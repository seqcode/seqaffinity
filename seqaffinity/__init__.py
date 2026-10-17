"""Whole-sequence PWM affinity scoring.

Five scores from one per-window energy array: max-site log-odds, total binding
affinity, length-normalised TBA, TRAP occupancy and a partition function with
steric exclusion. They are available per sequence (:func:`score_sequences`) and
genome-wide over sliding windows (:func:`scan_genome`).
"""

from .api import score_one, score_sequences
from .params import trap_r0
from .pwm import PWM, EnergyMatrix, load_pwm, to_energy
from .scan import scan_genome

__all__ = [
    "PWM",
    "EnergyMatrix",
    "load_pwm",
    "to_energy",
    "score_sequences",
    "score_one",
    "scan_genome",
    "trap_r0",
]
__version__ = "0.1.0"
