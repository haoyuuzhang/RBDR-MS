"""Resource configuration for Hierarchical FJSP.

Service Units:
  U1: M1_U1, M2_U1, M3_U1  (3 machines, full capability)
  U2: M1_U2, M2_U2         (2 machines, no M3)
"""

from typing import Dict, List

SERVICE_UNITS: Dict[str, List[str]] = {
    'U1': ['M1_U1', 'M2_U1', 'M3_U1'],
    'U2': ['M1_U2', 'M2_U2'],
}

ALL_MACHINES: List[str] = [m for ml in SERVICE_UNITS.values() for m in ml]

MACHINE_UNIT: Dict[str, str] = {}
for unit, machines in SERVICE_UNITS.items():
    for m in machines:
        MACHINE_UNIT[m] = unit


def machine_type_of(machine: str) -> str:
    """Extract machine type from machine name, e.g. 'M1_U2' -> 'M1'."""
    return machine.split('_')[0]


TRANSPORT_TIME = 2.0    # cross-unit transport time
BIG_M = 10000.0
