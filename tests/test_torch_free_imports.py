"""Guard that the on-disk layout and download engine stay free of torch/ComfyUI.

These modules are imported by lightweight consumers (the worker's planning code, third-party tools) that
must not pay for a torch import. The check runs in a subprocess so it reflects a clean interpreter rather
than modules other tests may have already imported.
"""

from __future__ import annotations

import subprocess
import sys


def test_layout_and_engine_import_without_torch_or_comfy() -> None:
    """Importing on_disk_layout and download_engine must not drag in torch or ComfyUI."""
    code = (
        "import sys\n"
        "import horde_model_reference.on_disk_layout\n"
        "import horde_model_reference.download_engine\n"
        "heavy = [name for name in sys.modules if name == 'torch' or name.split('.')[0] in {'comfy', 'hordelib'}]\n"
        "assert not heavy, f'unexpected heavy imports: {heavy}'\n"
        "print('torch-free-ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "torch-free-ok" in result.stdout
