"""SimPy-based discrete-event simulation engine for dynamic FJSP.

Architecture
------------
The engine manages machine loops, job arrivals, and disruptions.
Scheduling decisions are delegated to a pluggable ``BaseStrategy``.
Two strategy paradigms are supported via the same interface:

*Schedule-driven* (MILP, RightShift, Initial Schedule) —
  The strategy pre-computes a full plan; ``select_operation`` returns the
  next planned operation for the calling machine.

*Rule-driven* (Dispatching rules) —
  The strategy scores the ready-queue on the fly whenever a machine becomes
  free.

Usage::

    from case_generator import generate_case
    jobs, machines, disruptions = generate_case(seed=42)

    strategy = FullMILPStrategy()
    engine = SimulationEngine(jobs, machines, strategy, disruptions)
    schedule = engine.run()
"""

from __future__ import annotations

import simpy
from typing import Dict, List, Optional, Tuple

from models import Job, Operation, ScheduleEntry, MachineConfig, Disruption

TRANSPORT_TIME = 2.0


class SimulationEngine:
    """Discrete-event simulation that drives job-shop execution.

    Parameters
    ----------
    jobs:          All jobs (including future arrivals).
    machines:      Machine definitions keyed by machine_id.
    strategy:      A ``BaseStrategy`` instance.
    disruptions:   Known disruptions applied during the run.
    """

    def __init__(
        self,
        jobs: List[Job],
        machines: Dict[str, MachineConfig],
        strategy,
        disruptions: Optional[List[Disruption]] = None,
    ):
        self.env = simpy.Environment()

        # --- static configuration ---
        self.jobs = sorted(jobs, key=lambda j: (j.arrival_time, j.job_id))
        self.machines = machines          # machine_id -> MachineConfig
        self.disruptions = disruptions or []
        self.strategy = strategy

        # --- mutable state ---
        # op_status[(job_id, op_idx)] -> 'pending' | 'ready' | 'in_progress' | 'completed'
        self.op_status: Dict[Tuple[int, int], str] = {}
        self.machine_busy_until: Dict[str, float] = {}  # machine_id -> end_time
        self.completed_entries: List[ScheduleEntry] = []
        self.visible_jobs: List[Job] = []
        self.machine_factors: Dict[str, float] = {m: 1.0 for m in machines}
        self._idle_rounds = 0
        self._max_idle_rounds = 20000  # safety: prevent infinite idle loops (covers ~10000 time units / 0.5s polling)
        self._job_last_unit: Dict[int, str] = {}  # job_id -> unit of last completed op

        # For tracking in-progress ops (used by disruption handler)
        self._current_op: Dict[str, Tuple[int, int, str, float, float]] = {}
        # machine_id -> (job_id, op_idx, unit, start_time, base_duration)

        # For handling release_date delays
        self._release_events_scheduled: set = set()

        # Pre-computed factor-change events for _adjusted_duration
        # machine_id -> [(time, new_factor), ...] sorted by time
        self._factor_events: Dict[str, List[Tuple[float, float]]] = {}
        for mid in machines:
            self._factor_events[mid] = [(0.0, 1.0)]
        for d in self.disruptions:
            events = self._factor_events.setdefault(d.machine_id, [(0.0, 1.0)])
            events.append((d.time, d.factor))
            if d.duration > 0:
                recovery_time = d.time + d.duration
                events.append((recovery_time, 1.0))
        for mid in self._factor_events:
            self._factor_events[mid].sort(key=lambda x: x[0])

        # --- initialization ---
        for job in self.jobs:
            for op in job.operations:
                self.op_status[(job.job_id, op.op_idx)] = 'pending'

        for job in self.jobs:
            if job.arrival_time <= 0:
                self.visible_jobs.append(job)
                if job.release_date <= 0:
                    self.op_status[(job.job_id, 0)] = 'ready'

        # Give strategy a back-reference
        strategy.attach(self)

    # ── public API ───────────────────────────────────────────────────────

    def run(self, until: Optional[float] = None) -> List[ScheduleEntry]:
        """Run the simulation and return the completed schedule.

        If *until* is None, defaults to ``time_horizon * 2`` as a hard safety cap
        (jobs and disruptions all occur within the horizon, so 2× is generous).
        """
        if until is None:
            # Derive a hard time cap from the latest disruption/job arrival
            max_arrival = max((j.arrival_time for j in self.jobs), default=0)
            max_release = max((j.release_date for j in self.jobs), default=0)
            max_disruption = max((d.time for d in self.disruptions), default=0)
            horizon = max(max_arrival, max_release, max_disruption, 200.0)
            until = horizon * 2.0
        # Start machine loops
        for mid in self.machines:
            self.env.process(self._machine_loop(mid))

        # Schedule future job arrivals
        for job in self.jobs:
            if job.arrival_time > 0:
                self.env.process(self._job_arrival_process(job))

        # Schedule release events for jobs that arrive before their release
        for job in self.jobs:
            if job.release_date > max(0, job.arrival_time):
                self.env.process(self._release_event(job))

        # Schedule disruptions
        for d in self.disruptions:
            self.env.process(self._disruption_process(d))

        self.env.run(until=until)
        return self.completed_entries

    # ── machine loop ─────────────────────────────────────────────────────

    def _machine_loop(self, machine_id: str):
        """Perpetual loop: wait until free, ask strategy, process, repeat."""
        cfg = self.machines[machine_id]

        while True:
            # Wait until machine is free
            busy_until = self.machine_busy_until.get(machine_id, 0)
            if busy_until > self.env.now:
                yield self.env.timeout(busy_until - self.env.now)

            # Complete the operation that just finished (if any)
            self._finalize_op(machine_id)

            if self._all_done_or_deadlocked():
                return

            # Get candidates for this machine
            candidates = self._get_ready_ops(machine_type=cfg.machine_type)

            result = self.strategy.select_operation(
                machine_id=machine_id,
                machine_type=cfg.machine_type,
                unit=cfg.unit,
                current_time=self.env.now,
                ready_ops=candidates,
            )

            if result is None:
                self._idle_rounds += 1
                if self._idle_rounds >= self._max_idle_rounds:
                    return
                yield self.env.timeout(0.5)
                continue

            self._idle_rounds = 0  # reset — productive round
            job_id, op_idx, unit_name, base_duration = result

            # Cross-unit transport delay (only when switching units)
            if op_idx > 0:
                prev_unit = self._job_last_unit.get(job_id)
                if prev_unit is not None and prev_unit != unit_name:
                    yield self.env.timeout(TRANSPORT_TIME)
                    # Re-check: another machine may have claimed this op
                    # during the transport delay
                    if self.op_status.get((job_id, op_idx)) != 'ready':
                        continue

            # Compute actual duration accounting for time-varying machine factors
            duration = self._adjusted_duration(machine_id, base_duration,
                                               self.env.now)

            # Mark operation in-progress
            start = self.env.now
            self.op_status[(job_id, op_idx)] = 'in_progress'
            self.machine_busy_until[machine_id] = start + duration
            self._current_op[machine_id] = (job_id, op_idx, unit_name,
                                            start, base_duration)

            yield self.env.timeout(duration)

    # ── job arrival / release ────────────────────────────────────────────

    def _job_arrival_process(self, job: Job):
        yield self.env.timeout(job.arrival_time - self.env.now)
        self.visible_jobs.append(job)
        if job.release_date <= self.env.now:
            self._activate_first_op(job)
        self.strategy.on_job_arrival(job)

    def _release_event(self, job: Job):
        """Fire at job.release_date to activate the first operation."""
        delay = job.release_date - self.env.now
        if delay > 0:
            yield self.env.timeout(delay)
        self._activate_first_op(job)

    def _activate_first_op(self, job: Job):
        key = (job.job_id, 0)
        if self.op_status.get(key) == 'pending':
            self.op_status[key] = 'ready'

    # ── disruption ───────────────────────────────────────────────────────

    def _disruption_process(self, disruption: Disruption):
        """Apply disruption at its onset time; schedule recovery if duration > 0."""
        delay = disruption.time - self.env.now
        if delay > 0:
            yield self.env.timeout(delay)

        self.machine_factors[disruption.machine_id] = disruption.factor
        self.strategy.on_disruption(disruption)

        # Schedule automatic recovery
        if disruption.duration > 0:
            yield self.env.timeout(disruption.duration)
            self.machine_factors[disruption.machine_id] = 1.0
            # Notify strategy of recovery as a "reverse disruption" (factor=1.0)
            recovery = Disruption(
                time=self.env.now,
                machine_id=disruption.machine_id,
                factor=1.0,
                duration=0.0,
                description=f"{disruption.machine_id} recovered",
            )
            self.strategy.on_disruption(recovery)

    # ── operation lifecycle ──────────────────────────────────────────────

    def _finalize_op(self, machine_id: str):
        """Move a just-completed operation to 'completed' and unlock successor."""
        cur = self._current_op.pop(machine_id, None)
        if cur is None:
            return
        job_id, op_idx, unit_name, start_time, _base_dur = cur
        end_time = self.env.now

        self.op_status[(job_id, op_idx)] = 'completed'
        self.machine_busy_until.pop(machine_id, None)
        self._job_last_unit[job_id] = unit_name

        entry = ScheduleEntry(job_id=job_id, op_idx=op_idx, machine=machine_id,
                              service_unit=unit_name, start_time=start_time,
                              end_time=end_time)
        self.completed_entries.append(entry)

        # Unlock successor immediately — transport delay is handled at start time
        job = self._find_job(job_id)
        if job and op_idx + 1 < len(job.operations):
            next_key = (job_id, op_idx + 1)
            if self.op_status.get(next_key) == 'pending':
                self.op_status[next_key] = 'ready'

        self.strategy.on_operation_complete(entry)

    # ── helpers ──────────────────────────────────────────────────────────

    def _get_ready_ops(self, machine_type: Optional[str] = None
                       ) -> List[Tuple[Job, Operation]]:
        """Return operations that are ready and (optionally) match machine_type."""
        result: List[Tuple[Job, Operation]] = []
        for job in self.visible_jobs:
            if job.release_date > self.env.now:
                continue
            for op in job.operations:
                key = (job.job_id, op.op_idx)
                status = self.op_status.get(key, 'pending')
                if status == 'ready':
                    if machine_type is None or op.machine_type == machine_type:
                        result.append((job, op))
                    break  # only first unprocessed op per job
                if status in ('pending', 'in_progress'):
                    break  # this op (and later ones) blocked
                # 'completed' → continue to next op
        return result

    def _adjusted_duration(self, machine_id: str, base_duration: float,
                           start_time: float) -> float:
        """Compute actual wall-clock duration accounting for time-varying factors.

        Walks through the pre-computed factor-change events for *machine_id*,
        accumulating ``base_duration`` units of "effective work" segment by
        segment.  Handles disruption onset AND recovery mid-operation.
        """
        events = self._factor_events.get(machine_id, [(0.0, 1.0)])

        remaining = base_duration  # effective work still needed
        seg_start = start_time

        # Find current segment index (the one containing start_time)
        seg_idx = 0
        for i, (t, _) in enumerate(events):
            if t <= start_time:
                seg_idx = i
            else:
                break

        while remaining > 1e-9 and seg_idx < len(events):
            _, seg_factor = events[seg_idx]
            seg_end = events[seg_idx + 1][0] if seg_idx + 1 < len(events) else float('inf')

            seg_available = seg_end - seg_start
            work_in_segment = seg_available / seg_factor  # base units completed

            if work_in_segment >= remaining:
                # Operation finishes within this segment
                return (seg_start + remaining * seg_factor) - start_time

            remaining -= work_in_segment
            seg_start = seg_end
            seg_idx += 1

        # Fell through all events — finish with current (last) factor
        current_factor = self.machine_factors.get(machine_id, 1.0)
        return (seg_start + remaining * current_factor) - start_time

    def _find_job(self, job_id: int) -> Optional[Job]:
        for j in self.visible_jobs:
            if j.job_id == job_id:
                return j
        for j in self.jobs:
            if j.job_id == job_id:
                return j
        return None

    def _all_done(self) -> bool:
        """Check whether every operation of every job is completed."""
        for job in self.jobs:
            for op in job.operations:
                if self.op_status.get((job.job_id, op.op_idx)) != 'completed':
                    return False
        return True

    def _all_done_or_deadlocked(self) -> bool:
        """Return True when simulation should stop (done or stuck)."""
        if self._all_done():
            return True
        # Deadlock: no machine is busy, nothing is ready
        any_busy = any(
            self.machine_busy_until.get(m, 0) > self.env.now
            for m in self.machines
        )
        if any_busy:
            return False
        if self._get_ready_ops():
            return False
        # Future jobs will arrive — keep waiting (prevents premature termination)
        for job in self.jobs:
            if job.arrival_time > self.env.now:
                return False
        # No progress possible — mark remaining ops as skipped
        for job in self.jobs:
            for op in job.operations:
                key = (job.job_id, op.op_idx)
                if self.op_status.get(key) == 'pending':
                    self.op_status[key] = 'completed'
        return True

    def get_state_snapshot(self) -> dict:
        """Lightweight state dict for strategies that need it."""
        ready_ops = self._get_ready_ops()
        return {
            'current_time': self.env.now,
            'visible_jobs': list(self.visible_jobs),
            'completed_entries': list(self.completed_entries),
            'machine_busy_until': dict(self.machine_busy_until),
            'machine_factors': dict(self.machine_factors),
            'op_status': dict(self.op_status),
            'ready_ops': [(j.job_id, o.op_idx) for j, o in ready_ops],
        }
