"""
Evaluation module — metrics computation and report generation.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.metrics import compute_all_metrics, print_report

# Re-export for backward compatibility
__all__ = ["compute_all_metrics", "print_report"]
