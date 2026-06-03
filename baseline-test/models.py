"""Data models shared across the baseline simulation framework."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Operation:
    job_id: int
    op_idx: int
    machine_type: str          # 'M1', 'M2', 'M3'
    unit_times: Dict[str, float]   # {unit_name: processing_time}


@dataclass
class Job:
    job_id: int
    release_date: float
    due_date: float
    alpha: float
    beta: float
    operations: List[Operation]
    arrival_time: float = 0.0


@dataclass
class ScheduleEntry:
    job_id: int
    op_idx: int
    machine: str
    service_unit: str
    start_time: float
    end_time: float
    fixed: bool = False

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class MachineConfig:
    machine_id: str      # 'M1_U1'
    machine_type: str    # 'M1'
    unit: str            # 'U1'


@dataclass
class Disruption:
    time: float              # when the disruption starts
    machine_id: str          # which machine is affected
    factor: float = 2.0      # processing-time multiplier during the disruption
    duration: float = 0.0    # how long it lasts (0 = permanent, i.e. no recovery)
    description: str = ""
