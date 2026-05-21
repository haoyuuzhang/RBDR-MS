"""Data classes for Hierarchical Flexible Job-Shop Scheduling."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Operation:
    """A single operation within a job."""
    job_id: int
    op_idx: int
    machine_type: str          # 'M1', 'M2', or 'M3'
    unit_times: Dict[str, float]   # {unit_name: processing_time}


@dataclass
class Job:
    """A job (order) composed of an ordered sequence of operations."""
    job_id: int
    release_date: float
    due_date: float
    alpha: float
    beta: float
    operations: List[Operation]
    arrival_time: float = 0.0


@dataclass
class ScheduleEntry:
    """Records the assignment and timing of one operation."""
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
