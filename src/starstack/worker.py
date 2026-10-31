"""Separate worker process; GUI disconnects do not cancel jobs."""
import sys
from .core import run_job

if __name__ == "__main__":
    run_job(sys.argv[1])

