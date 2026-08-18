"""pytest configuration for the fused_layernorm test suite.

* Registers the ``cuda`` marker (also declared in ``pyproject.toml``) so the
  suite runs cleanly even when pytest is invoked from a directory without that
  file.
* Seeds torch's RNG before every test so failures are reproducible.

The GPU tests in ``test_layernorm.py`` are skipped -- not failed -- when the
``fused_layernorm_cuda`` extension is not built or no CUDA device is present,
so ``python -m pytest`` exits 0 on a CPU-only machine.  Nothing here imports
the extension at collection time.
"""

from __future__ import annotations

import pytest
import torch


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "cuda: requires a CUDA GPU and the compiled fused_layernorm_cuda extension",
    )


@pytest.fixture(autouse=True)
def _seed_torch() -> None:
    """Make every test start from the same RNG state."""
    torch.manual_seed(0)
