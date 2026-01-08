""" "Pytest configuration file."""

import logging
import os
from pathlib import Path
import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Configure logging for pytest.

    Args:
        config (pytest.Config): The pytest configuration object.
    """

    tests_folder = Path(__file__).parent
    if tests_folder.name != "tests":
        raise RuntimeError("The conftest.py file must be located in the 'tests' directory.")

    log_dir = tests_folder / "logs"
    log_dir.mkdir(exist_ok=True)

    pid = os.getpid()
    log_path = log_dir / f"pytest-{pid}.log"

    logging.basicConfig(
        level=logging.INFO,
        format=("%(asctime)s | %(levelname)-7s | %(process)d | %(name)s | %(message)s"),
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
        force=True,
    )

    logging.getLogger("qiskit").setLevel(logging.WARNING)
