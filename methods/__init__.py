"""Method entrypoints for pipeline runners"""

from .scanpy import run_scanpy
from .rapids import run_rapids

__all__ = ["run_scanpy", "run_rapids"]
