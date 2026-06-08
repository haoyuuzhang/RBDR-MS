"""Genetic Algorithm solver for Flexible Job-Shop Scheduling Problem (FJSP).

Provides a GA-based alternative to the Gurobi MILP solver, usable both as a
standalone global scheduler and as a lower-level unit solver in the bi-level
architecture.

Encoding
--------
Two-vector chromosome (Gen, Tsujimura & Kubota 1994):
  OS — Operation Sequence (permutation with repetitions of job IDs)
  MA — Machine Assignment   (dict: (job_id, op_idx) → machine_name)

The OS respects job precedence implicitly: the *k*-th occurrence of job *j*
always refers to the *k*-th unfixed operation of that job.  Fixed (already
committed or completed) operations are skipped during decoding.

Usage
-----
Global GA (same signature as :func:`scheduler.schedule_makespan_milp`):

    from ga_scheduler import schedule_makespan_ga
    schedule = schedule_makespan_ga(jobs, fixed_entries, current_time=0.0,
                                    time_limit=120.0)

Bi-level unit solver:

    from ga_scheduler import GAUnitSolver
    solver = GAUnitSolver(unit_machines=['M1','M2','M3','M4'])
    schedule = solver.solve(unit_jobs, fixed_entries, current_time=2.0)
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

from models import Job, Operation, ScheduleEntry
from bi_level_scheduler import UnitSolver

# ═══════════════════════════════════════════════════════════════════════════════
#  Chromosome
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Chromosome:
    """Two-vector chromosome for the FJSP.

    Attributes
    ----------
    os : list[int]
        Operation Sequence.  Job *j* appears ``n_j`` times; its *k*-th
        occurrence maps to the *k*-th **unfixed** operation of job *j*.
    ma : dict
        Machine Assignment: ``{(job_id, op_idx): machine_name}`` for every
        unfixed operation.
    """
    os: List[int]
    ma: Dict[Tuple[int, int], str]

    def copy(self) -> 'Chromosome':
        return Chromosome(list(self.os), dict(self.ma))


# ═══════════════════════════════════════════════════════════════════════════════
#  GA solver
# ═══════════════════════════════════════════════════════════════════════════════

class FJSPGaSolver:
    """Genetic Algorithm for FJSP makespan minimisation.

    Parameters
    ----------
    pop_size : int
        Population size (default 300 for global, 150 for unit-level).
    n_generations : int
        Maximum number of generations.
    tournament_size : int
        Tournament selection pressure.
    crossover_rate : float
        Probability of applying crossover to a pair of parents.
    mutation_rate_os : float
        Per-gene probability of OS mutation (swap adjacent).
    mutation_rate_ma : float
        Per-gene probability of MA mutation (random reassign).
    elite_size : int
        Number of best chromosomes copied unchanged each generation.
    patience : int
        Early-stop if no improvement for this many generations.
    time_limit : float
        Maximum wall-clock time in seconds.
    unit_machines : list[str] or None
        If set, only schedule operations that have at least one feasible
        machine in this list.  Used in bi-level mode to restrict a unit
        solver to its own machine pool.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Print progress messages.
    """

    def __init__(
        self,
        pop_size: int = 300,
        n_generations: int = 500,
        tournament_size: int = 3,
        crossover_rate: float = 0.90,
        mutation_rate_os: float = 0.10,
        mutation_rate_ma: float = 0.15,
        elite_size: int = 3,
        patience: int = 100,
        time_limit: float = 120.0,
        unit_machines: Optional[List[str]] = None,
        seed: int = 42,
        verbose: bool = True,
        local_search: bool = True,
        n_restarts: int = 1,
    ):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate_os = mutation_rate_os
        self.mutation_rate_ma = mutation_rate_ma
        self.elite_size = max(1, elite_size)
        self.patience = patience
        self.time_limit = time_limit
        self.unit_machines = set(unit_machines) if unit_machines else None
        self.seed = seed
        self.verbose = verbose
        self.local_search = local_search
        self.n_restarts = max(1, n_restarts)

        random.seed(seed)
        self.rng = random.Random(seed)

        # Set during solve()
        self._jobs: List[Job] = []
        self._job_map: Dict[int, Job] = {}
        self._fixed_keys: Set[Tuple[int, int]] = set()
        self._fixed_entries: List[ScheduleEntry] = []
        self._current_time: float = 0.0
        self._machine_deadlines: Dict[str, float] = {}
        self._unfixed_keys: List[Tuple[int, int]] = []       # all unfixed ops (canonical order)
        self._unfixed_per_job: Dict[int, List[int]] = {}      # job_id → [op_idx, ...] (sorted)
        self._all_machines: List[str] = []
        self._infeasible_penalty: float = 0.0
        self._unfixed_total: int = 0  # total number of unfixed operations

    # ── Public entry point ───────────────────────────────────────────────────

    def solve(
        self,
        jobs: List[Job],
        fixed_entries: List[ScheduleEntry],
        current_time: float = 0.0,
        machine_deadlines: Optional[Dict[str, float]] = None,
    ) -> Tuple[Optional[List[ScheduleEntry]], dict]:
        """Run the GA and return (best_schedule, stats_dict).

        Uses multi-start with different random seeds when ``n_restarts > 1``.
        Each restart seeds an independent GA run; the best result across all
        restarts is returned.

        Returns ``(None, stats)`` if no feasible schedule is found.
        """
        t_start = time.perf_counter()
        total_evals = 0

        self._jobs = jobs
        self._job_map = {j.job_id: j for j in jobs}
        self._fixed_entries = list(fixed_entries)
        self._fixed_keys = {(e.job_id, e.op_idx) for e in fixed_entries}
        self._current_time = current_time
        self._machine_deadlines = dict(machine_deadlines) if machine_deadlines else {}

        self._build_op_index()

        # No unfixed ops — everything already scheduled
        if not self._unfixed_keys:
            elapsed = time.perf_counter() - t_start
            return list(fixed_entries), {
                'best_fitness': self._compute_makespan_from_fixed(),
                'generations': 0,
                'compute_time': elapsed,
                'evaluations': 0,
            }

        self._unfixed_total = sum(len(v) for v in self._unfixed_per_job.values())
        self._all_machines = self._collect_machines()
        self._infeasible_penalty = self._compute_big_m()

        # ── Multi-start loop ──────────────────────────────────────────────
        best_overall: Optional[Chromosome] = None
        best_overall_fitness = float('inf')

        for restart in range(self.n_restarts):
            # Use a different seed for each restart
            if self.n_restarts > 1:
                self.rng = random.Random(self.seed + restart * 1000)

            remaining = self.time_limit - (time.perf_counter() - t_start)
            if remaining <= 1.0:
                break

            best_chromosome, best_fitness, gen, evals = self._run_single_ga(
                time_budget=remaining)
            total_evals += evals

            if best_fitness < best_overall_fitness - 1e-9:
                best_overall_fitness = best_fitness
                best_overall = best_chromosome.copy()
                if self.verbose and self.n_restarts > 1:
                    print(f"    GA restart {restart + 1}/{self.n_restarts}: "
                          f"new best = {best_fitness:.3f}")

        # Decode best chromosome
        if best_overall is None:
            return None, {'best_fitness': float('inf'), 'generations': 0,
                          'compute_time': time.perf_counter() - t_start,
                          'evaluations': total_evals}

        schedule = self._decode_to_schedule(best_overall)
        elapsed = time.perf_counter() - t_start

        if self.verbose:
            cmax = max((e.end_time for e in schedule), default=0.0)
            print(f"    GA: final C_max = {cmax:.3f}  "
                  f"(time = {elapsed:.1f}s,  evals = {total_evals})")

        stats = {
            'best_fitness': best_overall_fitness,
            'generations': gen,
            'compute_time': elapsed,
            'evaluations': total_evals,
        }

        return schedule, stats

    def _run_single_ga(
        self, time_budget: float
    ) -> Tuple[Chromosome, float, int, int]:
        """Run a single GA evolution (one restart).

        Returns (best_chromosome, best_fitness, generations, evaluations).
        """
        t_start = time.perf_counter()
        total_evals = 0

        # Initialise population with heuristic seeds
        pop = self._initialize_population()
        fitnesses = [self._evaluate(ind) for ind in pop]
        total_evals += len(pop)

        # Apply local search to initial elite
        if self.local_search:
            sorted_idx = sorted(range(len(pop)), key=lambda i: fitnesses[i])
            for i in range(min(self.elite_size, len(pop))):
                improved_fit, improved_ch = self._local_search(pop[sorted_idx[i]])
                total_evals += 1  # approximate
                if improved_fit < fitnesses[sorted_idx[i]] - 1e-9:
                    pop[sorted_idx[i]] = improved_ch
                    fitnesses[sorted_idx[i]] = improved_fit

        best_idx = min(range(len(pop)), key=lambda i: fitnesses[i])
        best_fitness = fitnesses[best_idx]
        best_chromosome = pop[best_idx].copy()
        generations_no_improve = 0

        # Evolution loop
        for gen in range(self.n_generations):
            if time.perf_counter() - t_start > time_budget:
                if self.verbose:
                    print(f"    GA: time budget reached at gen {gen}")
                break

            # Build next generation
            new_pop: List[Chromosome] = []

            # Elitism
            sorted_idx = sorted(range(len(pop)), key=lambda i: fitnesses[i])
            for i in range(self.elite_size):
                new_pop.append(pop[sorted_idx[i]].copy())

            # Fill rest with selection + crossover + mutation
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(pop, fitnesses)
                p2 = self._tournament_select(pop, fitnesses)

                if self.rng.random() < self.crossover_rate:
                    child_os = self._crossover_os(p1.os, p2.os)
                    child_ma = self._crossover_ma(p1.ma, p2.ma)
                else:
                    child_os = list(p1.os)
                    child_ma = dict(p1.ma)

                self._mutate_os(child_os)
                self._mutate_ma(child_ma)

                new_pop.append(Chromosome(child_os, child_ma))

            pop = new_pop[:self.pop_size]
            fitnesses = [self._evaluate(ind) for ind in pop]
            total_evals += len(pop)

            curr_best_idx = min(range(len(pop)), key=lambda i: fitnesses[i])
            curr_best = fitnesses[curr_best_idx]

            if curr_best < best_fitness - 1e-9:
                best_fitness = curr_best
                best_chromosome = pop[curr_best_idx].copy()
                generations_no_improve = 0

                # Apply local search to new best
                if self.local_search and self._unfixed_total <= 50:
                    improved_fit, improved_ch = self._local_search(
                        best_chromosome)
                    if improved_fit < best_fitness - 1e-9:
                        best_fitness = improved_fit
                        best_chromosome = improved_ch
            else:
                generations_no_improve += 1

            if self.verbose and gen % 50 == 0:
                print(f"    GA gen {gen:4d}  best = {best_fitness:.3f}  "
                      f"curr = {curr_best:.3f}")

            # Diversity injection when stuck
            if generations_no_improve >= self.patience // 2 and generations_no_improve % 20 == 0:
                n_inject = max(1, self.pop_size // 5)
                for i in range(n_inject):
                    idx = self.rng.randrange(self.elite_size, len(pop))
                    pop[idx] = Chromosome(self._heuristic_os(), self._greedy_ma())
                    fitnesses[idx] = self._evaluate(pop[idx])
                    total_evals += 1

            if generations_no_improve >= self.patience:
                if self.verbose:
                    print(f"    GA: converged at gen {gen}")
                break

        return best_chromosome, best_fitness, gen + 1, total_evals

    # ── Internal: index building ─────────────────────────────────────────────

    def _build_op_index(self) -> None:
        """Map unfixed operations per job, in order."""
        self._unfixed_per_job = {}
        self._unfixed_keys = []

        for job in self._jobs:
            unfixed = []
            for op in job.operations:
                key = (job.job_id, op.op_idx)
                if key not in self._fixed_keys:
                    # Check if this op is feasible on unit machines (if restricted)
                    if self.unit_machines is not None:
                        if not any(m in self.unit_machines for m in op.times):
                            continue  # not this unit's op → skip
                    unfixed.append(op.op_idx)
                    self._unfixed_keys.append(key)
            if unfixed:
                self._unfixed_per_job[job.job_id] = sorted(unfixed)

    def _collect_machines(self) -> List[str]:
        """All machines referenced in jobs, fixed entries, and deadlines."""
        machines: Set[str] = set()
        for job in self._jobs:
            for op in job.operations:
                machines.update(op.times.keys())
        for fe in self._fixed_entries:
            machines.add(fe.machine)
        machines.update(self._machine_deadlines.keys())
        return sorted(machines)

    def _compute_big_m(self) -> float:
        """Large penalty value for infeasible schedules."""
        total = 0.0
        for job in self._jobs:
            for op in job.operations:
                if op.times:
                    total += max(op.times.values())
        return max(total * 2.0, 1000.0)

    def _compute_makespan_from_fixed(self) -> float:
        """Makespan of the fixed schedule alone."""
        return max((e.end_time for e in self._fixed_entries), default=0.0)

    # ── Internal: initialisation ─────────────────────────────────────────────

    def _initialize_population(self) -> List[Chromosome]:
        """Create initial population with diverse heuristic seeding.

        Mix of OS generation methods and MA strategies:
        - 25%: random OS + random MA
        - 20%: heuristic OS (FIFO/SPT varied) + greedy MA
        - 20%: random OS + greedy MA (SPT machine)
        - 20%: random OS + workload-balanced MA
        - 15%: heuristic OS + random MA
        """
        pop: List[Chromosome] = []
        n = self.pop_size

        for i in range(n):
            if i < n * 0.25:
                os = self._random_os()
                ma = self._random_ma()
            elif i < n * 0.45:
                os = self._heuristic_os()
                ma = self._greedy_ma()
            elif i < n * 0.65:
                os = self._random_os()
                ma = self._greedy_ma()
            elif i < n * 0.85:
                os = self._random_os()
                ma = self._balanced_ma()
            else:
                os = self._heuristic_os()
                ma = self._random_ma()
            pop.append(Chromosome(os, ma))

        return pop

    def _random_os(self) -> List[int]:
        """Generate a random operation sequence."""
        seq: List[int] = []
        for jid in sorted(self._unfixed_per_job.keys()):
            seq.extend([jid] * len(self._unfixed_per_job[jid]))
        self.rng.shuffle(seq)
        return seq

    def _heuristic_os(self) -> List[int]:
        """Generate an operation sequence using a randomly-chosen heuristic.

        Heuristics include:
          - FIFO: jobs in order of arrival then op order
          - SPT: operations sorted by minimum feasible processing time
          - MWKR: jobs sorted by total remaining work (descending)
          - Random-biased: FIFO with random swaps
        """
        method = self.rng.choice(['fifo', 'spt', 'mwkr', 'fifo_biased'])

        if method == 'fifo':
            seq = []
            for jid in sorted(self._job_map.keys()):
                if jid in self._unfixed_per_job:
                    seq.extend([jid] * len(self._unfixed_per_job[jid]))
            return seq

        elif method == 'spt':
            # Collect (job_id, op_idx, min_proc_time) for all unfixed ops
            ops_info = []
            for jid, op_indices in self._unfixed_per_job.items():
                for oidx in op_indices:
                    op = self._job_map[jid].operations[oidx]
                    min_t = min(op.times.values()) if op.times else float('inf')
                    ops_info.append((jid, oidx, min_t))
            ops_info.sort(key=lambda x: x[2])  # shortest first
            seq = [jid for jid, oidx, _ in ops_info]
            return seq

        elif method == 'mwkr':
            # Most Work Remaining: sum of min processing times of remaining ops
            job_work = {}
            for jid, op_indices in self._unfixed_per_job.items():
                total = 0.0
                for oidx in op_indices:
                    op = self._job_map[jid].operations[oidx]
                    total += min(op.times.values()) if op.times else 0
                job_work[jid] = total
            sorted_jobs = sorted(job_work.keys(), key=lambda j: job_work[j],
                                 reverse=True)
            seq = []
            for jid in sorted_jobs:
                seq.extend([jid] * len(self._unfixed_per_job[jid]))
            return seq

        else:  # fifo_biased
            seq = []
            for jid in sorted(self._job_map.keys()):
                if jid in self._unfixed_per_job:
                    seq.extend([jid] * len(self._unfixed_per_job[jid]))
            # Apply random swaps
            for _ in range(len(seq) // 3):
                i = self.rng.randrange(len(seq))
                j = self.rng.randrange(len(seq))
                if seq[i] != seq[j]:
                    seq[i], seq[j] = seq[j], seq[i]
            return seq

    def _random_ma(self) -> Dict[Tuple[int, int], str]:
        """Random feasible machine for each unfixed operation."""
        ma: Dict[Tuple[int, int], str] = {}
        for key in self._unfixed_keys:
            jid, oidx = key
            op = self._job_map[jid].operations[oidx]
            feasible = self._feasible_machines_for_op(op)
            if feasible:
                ma[key] = self.rng.choice(feasible)
        return ma

    def _greedy_ma(self) -> Dict[Tuple[int, int], str]:
        """Shortest-processing-time (SPT) machine for each unfixed operation."""
        ma: Dict[Tuple[int, int], str] = {}
        for key in self._unfixed_keys:
            jid, oidx = key
            op = self._job_map[jid].operations[oidx]
            feasible = self._feasible_machines_for_op(op)
            if feasible:
                ma[key] = min(feasible, key=lambda m: op.times[m])
        return ma

    def _balanced_ma(self) -> Dict[Tuple[int, int], str]:
        """Workload-balanced machine assignment.

        Greedily assigns each operation to the feasible machine with the
        lightest current load (sum of processing times assigned so far).
        """
        ma: Dict[Tuple[int, int], str] = {}
        machine_load: Dict[str, float] = {
            m: 0.0 for m in self._all_machines}
        # Initialise load from fixed_entries
        for fe in self._fixed_entries:
            if fe.machine in machine_load:
                machine_load[fe.machine] += fe.duration

        # Process ops in a heuristic order (by job then op_idx)
        sorted_keys = sorted(self._unfixed_keys, key=lambda k: (k[0], k[1]))
        for key in sorted_keys:
            jid, oidx = key
            op = self._job_map[jid].operations[oidx]
            feasible = self._feasible_machines_for_op(op)
            if feasible:
                chosen = min(feasible, key=lambda m: machine_load.get(m, 0.0))
                ma[key] = chosen
                machine_load[chosen] = machine_load.get(chosen, 0.0) + op.times[chosen]
        return ma

    def _feasible_machines_for_op(self, op: Operation) -> List[str]:
        """Feasible machines for *op*, respecting unit restriction."""
        machines = list(op.times.keys())
        if self.unit_machines is not None:
            machines = [m for m in machines if m in self.unit_machines]
        return machines

    # ── Internal: evaluation (decoding) ──────────────────────────────────────

    def _evaluate(self, chromosome: Chromosome) -> float:
        """Decode chromosome and return makespan (lower is better).

        Infeasible schedules receive a large penalty.
        """
        makespan, _ = self._decode(chromosome)
        return makespan

    def _decode(self, chromosome: Chromosome) -> Tuple[float, List[ScheduleEntry]]:
        """Active schedule decoding.

        Returns
        -------
        (makespan, list_of_new_schedule_entries)
            *makespan* may include the penalty for unscheduled operations.
        """
        # ── State ─────────────────────────────────────────────────────────
        # machine_free_at[m] = earliest time machine m becomes free
        machine_free_at: Dict[str, float] = {
            m: self._current_time for m in self._all_machines
        }
        # job_ready_at[jid] = earliest time the next operation of job jid can start
        job_ready_at: Dict[int, float] = {}
        for j in self._jobs:
            job_ready_at[j.job_id] = max(self._current_time, j.arrival_time)

        # Apply fixed entries
        for fe in self._fixed_entries:
            if fe.machine in machine_free_at:
                machine_free_at[fe.machine] = max(
                    machine_free_at[fe.machine], fe.end_time)
            job_ready_at[fe.job_id] = max(
                job_ready_at.get(fe.job_id, 0.0), fe.end_time)

        # ── Decode OS ─────────────────────────────────────────────────────
        occurrence: Dict[int, int] = {jid: 0 for jid in self._unfixed_per_job}
        new_entries: List[ScheduleEntry] = []
        penalty = 0.0

        for jid in chromosome.os:
            if jid not in self._unfixed_per_job:
                continue

            k = occurrence[jid]
            occurrence[jid] += 1

            if k >= len(self._unfixed_per_job[jid]):
                continue

            op_idx = self._unfixed_per_job[jid][k]
            key = (jid, op_idx)

            if key in self._fixed_keys:
                continue

            op = self._job_map[jid].operations[op_idx]

            # ── Choose machine ──────────────────────────────────────────
            assigned_machine = chromosome.ma.get(key)
            feasible = self._feasible_machines_for_op(op)

            if not feasible:
                # No feasible machines: skip with penalty
                penalty += self._infeasible_penalty
                continue

            if assigned_machine not in feasible:
                # Repair: pick the shortest feasible machine
                assigned_machine = min(feasible, key=lambda m: op.times[m])

            proc_time = op.times[assigned_machine]

            # ── Compute earliest start ───────────────────────────────────
            start = max(job_ready_at[jid], machine_free_at[assigned_machine])
            end = start + proc_time

            # ── Deadline check ───────────────────────────────────────────
            deadline = self._machine_deadlines.get(assigned_machine, float('inf'))
            if end > deadline + 1e-9:
                # Try other feasible machines that can meet deadline
                candidates = []
                for m in feasible:
                    p = op.times[m]
                    dl = self._machine_deadlines.get(m, float('inf'))
                    cand_start = max(job_ready_at[jid], machine_free_at[m])
                    cand_end = cand_start + p
                    if cand_end <= dl + 1e-9:
                        candidates.append((m, cand_start, cand_end, p))
                if candidates:
                    # Pick the one with earliest completion
                    candidates.sort(key=lambda x: x[2])
                    assigned_machine, start, end, proc_time = candidates[0]
                else:
                    # No feasible machine meets deadline → penalty
                    penalty += self._infeasible_penalty
                    # Still schedule on original machine to maintain continuity
                    # (the penalty will drive evolution away from this choice)

            # ── Record ───────────────────────────────────────────────────
            new_entries.append(ScheduleEntry(
                job_id=jid, op_idx=op_idx, machine=assigned_machine,
                start_time=start, end_time=end))

            machine_free_at[assigned_machine] = end
            job_ready_at[jid] = end

        # ── Compute makespan ────────────────────────────────────────────
        all_ends = [e.end_time for e in self._fixed_entries]
        all_ends.extend(e.end_time for e in new_entries)
        makespan = max(all_ends) if all_ends else self._current_time

        return makespan + penalty, new_entries

    def _decode_to_schedule(self, chromosome: Chromosome) -> List[ScheduleEntry]:
        """Decode chromosome and return full schedule (fixed + new entries)."""
        _, new_entries = self._decode(chromosome)
        return list(self._fixed_entries) + new_entries

    # ── Internal: local search ────────────────────────────────────────────────

    def _local_search(self, chromosome: Chromosome) -> Tuple[float, Chromosome]:
        """Simple hill-climbing on a chromosome.

        Alternates between OS perturbations (adjacent swaps) and MA
        perturbations (single-machine reassignments).  Accepts the first
        improving move found; repeats until no improvement.

        Only applied when the chromosome has ≤ 50 unfixed operations
        (keeps per-evaluation cost low).
        """
        best_ch = chromosome.copy()
        best_fit = self._evaluate(best_ch)
        n = len(best_ch.os)

        if n < 2:
            return best_fit, best_ch

        improved = True
        max_iters = 10  # limit iterations to prevent stalling
        iteration = 0

        while improved and iteration < max_iters:
            improved = False
            iteration += 1

            # ── Phase 1: Try adjacent swaps in OS ─────────────────────────
            # Shuffle the order of attempts for diversity
            positions = list(range(n - 1))
            self.rng.shuffle(positions)
            for i in positions:
                if best_ch.os[i] == best_ch.os[i + 1]:
                    continue  # swapping same-job ops has no effect
                best_ch.os[i], best_ch.os[i + 1] = best_ch.os[i + 1], best_ch.os[i]
                new_fit = self._evaluate(best_ch)
                if new_fit < best_fit - 1e-9:
                    best_fit = new_fit
                    improved = True
                    break
                else:
                    best_ch.os[i], best_ch.os[i + 1] = best_ch.os[i + 1], best_ch.os[i]

            if improved:
                continue

            # ── Phase 2: Try machine reassignments ─────────────────────────
            ma_keys = list(best_ch.ma.keys())
            self.rng.shuffle(ma_keys)
            for key in ma_keys:
                jid, oidx = key
                op = self._job_map[jid].operations[oidx]
                feasible = self._feasible_machines_for_op(op)
                if len(feasible) < 2:
                    continue
                current = best_ch.ma[key]
                for new_m in feasible:
                    if new_m == current:
                        continue
                    old_m = best_ch.ma[key]
                    best_ch.ma[key] = new_m
                    new_fit = self._evaluate(best_ch)
                    if new_fit < best_fit - 1e-9:
                        best_fit = new_fit
                        improved = True
                        break
                    else:
                        best_ch.ma[key] = old_m
                if improved:
                    break

        return best_fit, best_ch

    # ── Internal: selection ──────────────────────────────────────────────────

    def _tournament_select(
        self, pop: List[Chromosome], fitnesses: List[float]
    ) -> Chromosome:
        """Binary tournament selection (lower fitness wins)."""
        candidates = self.rng.sample(range(len(pop)), k=self.tournament_size)
        best = min(candidates, key=lambda i: fitnesses[i])
        return pop[best]

    # ── Internal: crossover ──────────────────────────────────────────────────

    def _crossover_os(self, os1: List[int], os2: List[int]) -> List[int]:
        """Precedence-Preserving Order-based Crossover (POX) for OS.

        1. Randomly partition job IDs into two sets J1, J2.
        2. Child inherits positions of J1 jobs from parent 1.
        3. Remaining positions filled with J2 jobs in the order they appear
           in parent 2.
        """
        all_jids = list(self._unfixed_per_job.keys())
        if len(all_jids) <= 1:
            return list(os1)

        # Partition jobs
        self.rng.shuffle(all_jids)
        split = max(1, len(all_jids) // 2)
        j1_set = set(all_jids[:split])

        child = [0] * len(os1)
        # Place J1 jobs from parent 1 at same positions
        for i, jid in enumerate(os1):
            if jid in j1_set:
                child[i] = jid

        # Fill J2 jobs from parent 2 (in order)
        p2_iter = iter(jid for jid in os2 if jid not in j1_set)
        for i in range(len(child)):
            if child[i] == 0:
                child[i] = next(p2_iter)

        return child

    def _crossover_ma(
        self, ma1: Dict[Tuple[int, int], str], ma2: Dict[Tuple[int, int], str]
    ) -> Dict[Tuple[int, int], str]:
        """Uniform crossover for machine assignment.

        For each operation key, child inherits from either parent with equal
        probability.
        """
        child_ma: Dict[Tuple[int, int], str] = {}
        for key in ma1:
            if key not in ma2:
                child_ma[key] = ma1[key]
            elif self.rng.random() < 0.5:
                child_ma[key] = ma1[key]
            else:
                child_ma[key] = ma2[key]
        return child_ma

    # ── Internal: mutation ───────────────────────────────────────────────────

    def _mutate_os(self, os: List[int]) -> None:
        """Swap mutation on OS: swap two positions with different job IDs."""
        for i in range(len(os)):
            if self.rng.random() < self.mutation_rate_os:
                j = self.rng.randrange(len(os))
                if os[i] != os[j]:
                    os[i], os[j] = os[j], os[i]

    def _mutate_ma(self, ma: Dict[Tuple[int, int], str]) -> None:
        """Random reassignment for MA: each operation has a chance to be
        reassigned to a different feasible machine."""
        for key in list(ma.keys()):
            if self.rng.random() < self.mutation_rate_ma:
                jid, oidx = key
                op = self._job_map[jid].operations[oidx]
                feasible = self._feasible_machines_for_op(op)
                if len(feasible) >= 2:
                    current = ma[key]
                    others = [m for m in feasible if m != current]
                    if others:
                        ma[key] = self.rng.choice(others)


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience function (matching scheduler.schedule_makespan_milp signature)
# ═══════════════════════════════════════════════════════════════════════════════

def schedule_makespan_ga(
    jobs: List[Job],
    fixed_entries: List[ScheduleEntry],
    current_time: float = 0.0,
    time_factors: Optional[Dict[str, float]] = None,
    machine_deadlines: Optional[Dict[str, float]] = None,
    time_limit: float = 120.0,
    pop_size: int = 300,
    n_generations: int = 500,
    seed: int = 42,
    verbose: bool = True,
    unit_machines: Optional[List[str]] = None,
) -> Optional[List[ScheduleEntry]]:
    """Solve FJSP via Genetic Algorithm (same signature style as the MILP).

    .. note::
        ``time_factors`` is accepted for signature compatibility but is
        **not yet implemented** in the GA decoder.  Pass ``None``.

    Returns
    -------
    list of ScheduleEntry  or  None if no feasible schedule found.
    """
    if time_factors:
        # The GA decoder currently does not support per-machine time scaling.
        # Raise or warn depending on strictness preference.
        import warnings
        warnings.warn("time_factors is not supported by the GA solver; ignored.")

    solver = FJSPGaSolver(
        pop_size=pop_size,
        n_generations=n_generations,
        time_limit=time_limit,
        unit_machines=unit_machines,
        seed=seed,
        verbose=verbose,
    )
    schedule, _stats = solver.solve(
        jobs, fixed_entries, current_time, machine_deadlines,
    )
    return schedule


# ═══════════════════════════════════════════════════════════════════════════════
#  GA-based UnitSolver for bi-level scheduling
# ═══════════════════════════════════════════════════════════════════════════════

class GAUnitSolver(UnitSolver):
    """Unit-level solver backed by a Genetic Algorithm.

    Implements the :class:`~bi_level_scheduler.UnitSolver` interface so it can
    be plugged into :func:`~bi_level_scheduler.simulate_bi_level` (or its GA
    equivalent) as a drop-in replacement for :class:`GurobiUnitSolver`.

    Parameters
    ----------
    unit_machines : list[str]
        Machines belonging to this service unit (e.g. ``['M1','M2','M3','M4']``).
        The GA will **only** schedule operations that have at least one feasible
        machine in this set.
    pop_size : int
        GA population size (default 150 for unit-level).
    n_generations : int
        Maximum generations (default 300).
    time_limit : float
        Solver time limit in seconds.
    seed : int
        Random seed.
    verbose : bool
        Print progress.

    Example
    -------
        solver_u1 = GAUnitSolver(['M1','M2','M3','M4'], pop_size=150, n_generations=200)
        schedule = solver_u1.solve(unit_jobs, fixed_entries, current_time=2.0,
                                   machine_deadlines={'M3': 6.0})
    """

    def __init__(
        self,
        unit_machines: List[str],
        pop_size: int = 150,
        n_generations: int = 300,
        time_limit: float = 60.0,
        local_search: bool = True,
        n_restarts: int = 1,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.unit_machines = list(unit_machines)
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.time_limit = time_limit
        self.local_search = local_search
        self.n_restarts = n_restarts
        self.seed = seed
        self.verbose = verbose

    def solve(
        self,
        jobs: List[Job],
        fixed_entries: List[ScheduleEntry],
        current_time: float,
        machine_deadlines: Optional[Dict[str, float]] = None,
        time_limit: float = 120.0,  # accepted for interface compatibility
    ) -> Optional[List[ScheduleEntry]]:
        """Solve the scheduling problem for a single service unit using GA.

        Only operations that have at least one feasible machine in
        ``self.unit_machines`` will be scheduled.  Operations outside this
        unit are silently skipped.
        """
        solver = FJSPGaSolver(
            pop_size=self.pop_size,
            n_generations=self.n_generations,
            time_limit=min(self.time_limit, time_limit),
            unit_machines=self.unit_machines,
            local_search=self.local_search,
            n_restarts=self.n_restarts,
            seed=self.seed,
            verbose=self.verbose,
        )
        schedule, _stats = solver.solve(
            jobs, fixed_entries, current_time, machine_deadlines,
        )
        return schedule
