# ML / LLM GPU readiness checklist

Use this as the plan for an interactive `gpu_preflight.ipynb` notebook. Run each cell and retain its output before beginning an ML or LLM project.

## 1. Understand the components

- NVIDIA driver: needed to use the GPU.
- CUDA Toolkit/developer tools: needed only to compile CUDA/C++ or custom CUDA extensions.
- PyTorch CUDA wheel: needed for PyTorch GPU execution. Its runtime is bundled with the wheel; it does not need to exactly equal the Toolkit version shown by `nvcc`.

Do not install a standalone Toolkit merely for ordinary PyTorch training. The NVIDIA driver is required.

## 2. Check GPU, driver, and supported runtime

Run this in PowerShell:

~~~powershell
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
~~~

Record:

- [ ] GPU name and total VRAM
- [ ] Driver Version
- [ ] CUDA Version from `nvidia-smi`

The CUDA Version displayed by `nvidia-smi` is the highest CUDA runtime supported by the installed driver, not proof that a CUDA Toolkit is installed.

Notebook cell:

~~~python
import platform
import subprocess
import sys
print("Python:", sys.version)
print("Platform:", platform.platform())
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False).stdout)
~~~

If `nvidia-smi` fails or no GPU appears, install/update the NVIDIA driver and reboot. Installing a Toolkit alone does not fix a missing driver.

## 3. Check CUDA developer tools

Run in PowerShell:

~~~powershell
nvcc --version
where.exe nvcc
$env:CUDA_PATH
~~~

- [ ] `nvcc --version` succeeds: the CUDA Toolkit compiler is installed and on `PATH`.
- [ ] `nvidia-smi` works but `nvcc` is missing: normal for a PyTorch-only project. Install the Toolkit only if compiling a custom extension or writing CUDA/C++.

### Install the CUDA Toolkit (Windows)

Only do this when the project needs `nvcc` (for example, to compile a custom CUDA extension). A normal PyTorch CUDA wheel already includes its runtime and does not need this installation.

1. Open NVIDIA's [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads) page.
2. Select **Windows** -> **x86_64** -> your installed Windows version -> **exe (local)**. Choose a Toolkit version compatible with the NVIDIA driver recorded in Step 2; the newest Toolkit is not automatically required.
3. Download and run the installer. Select **Custom (Advanced)** if you want to review components; install the **CUDA Toolkit**. The display-driver component is optional when the existing NVIDIA driver is newer and working.
4. Complete installation and open a new PowerShell window (or reboot if requested).
5. Verify the compiler and installation location:

~~~powershell
nvcc --version
where.exe nvcc
$env:CUDA_PATH
Get-ChildItem $env:CUDA_PATH
~~~

If `nvcc` is still not found, add the Toolkit bin directory for the installed version to the user or system `PATH`, then open a new terminal. The usual location is `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v<version>\bin`.

### Notebook cell: compare Toolkit and driver compatibility

This compares the CUDA Toolkit version reported by `nvcc` with the maximum CUDA runtime version reported by `nvidia-smi`. A Toolkit version at or below that driver ceiling is the normal compatible case. This is a preflight check only: it does not prove that every third-party CUDA extension supports the selected PyTorch, compiler, and GPU architecture.

~~~python
import re
import shutil
import subprocess

def run(command):
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or f"Command failed: {command}")
    return completed.stdout

def parse_cuda_version(text, pattern, source):
    match = re.search(pattern, text)
    if not match:
        raise RuntimeError(f"Could not find a CUDA version in {source} output.")
    return tuple(map(int, match.group(1).split(".")))

assert shutil.which("nvcc"), "nvcc is unavailable; install the Toolkit or fix PATH."
smi_output = run(["nvidia-smi"])
nvcc_output = run(["nvcc", "--version"])

driver_cuda = parse_cuda_version(smi_output, r"CUDA Version:\s*(\d+\.\d+)", "nvidia-smi")
toolkit_cuda = parse_cuda_version(nvcc_output, r"release\s+(\d+\.\d+)", "nvcc")

print("Driver CUDA compatibility ceiling:", ".".join(map(str, driver_cuda)))
print("Installed CUDA Toolkit:", ".".join(map(str, toolkit_cuda)))

if toolkit_cuda <= driver_cuda:
    print("PASS: the installed Toolkit is within the driver's reported CUDA compatibility ceiling.")
else:
    print(
        "FAIL: the Toolkit is newer than the driver's reported CUDA compatibility ceiling. "
        "Update the NVIDIA driver or install a Toolkit version no newer than that ceiling."
    )
~~~

Notebook cell:

~~~python
import shutil
import subprocess
nvcc = shutil.which("nvcc")
print("nvcc path:", nvcc)
print(subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout if nvcc else "Toolkit not installed or not on PATH")
~~~

## 4. Choose the PyTorch CUDA wheel

1. Open the official [PyTorch Start Locally selector](https://pytorch.org/get-started/locally/).
2. Choose the current stable release, OS, Pip, Python, and a CUDA platform compatible with the driver.
3. Install matching `torch`, `torchvision`, and `torchaudio` releases from the same official index. Never mix CUDA indexes.
4. An RTX Nvidia Gpu needs a recent CUDA wheel. Current official options include CUDA 12.8 (`cu128`) and CUDA 13.0 (`cu130`); confirm the selector before installation.

| Index | Meaning |
| --- | --- |
| `https://download.pytorch.org/whl/cu126` | CUDA 12.6 wheel |
| `https://download.pytorch.org/whl/cu128` | CUDA 12.8 wheel |
| `https://download.pytorch.org/whl/cu130` | CUDA 13.0 wheel |
| `https://download.pytorch.org/whl/cpu` | CPU-only wheel; unsuitable for CUDA training |

`torch.Tensor` is included in `torch`; it is not a package to install. If “tensor” means TensorBoard, install `tensorboard` for logging. TensorBoard has no CUDA-specific version.

## 5. `pyproject.toml` / uv configuration

Replace `cu128` only after choosing an official wheel index. The shown Torch-family releases are a compatible set; check the [official version matrix](https://pytorch.org/get-started/previous-versions/) when upgrading.

~~~toml
[project]
name = "scripter"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "nemo-toolkit[asr]>=2.3.0",
    "transformers>=4.57.3",
    "torch==2.11.0",
    "torchvision==0.26.0",
    "torchaudio==2.11.0",
    "tensorboard>=2.20.0",
    "pydub>=0.25.1",
]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]
torchvision = [{ index = "pytorch-cu128" }]
torchaudio = [{ index = "pytorch-cu128" }]
~~~

~~~powershell
uv lock
uv sync
~~~

- [ ] Keep `explicit = true`, so ordinary dependencies come from PyPI.
- [ ] Point all three Torch-family packages to the same named index.
- [ ] Commit `pyproject.toml` and `uv.lock`.

For a new project:

~~~powershell
uv add torch torchvision torchaudio --index pytorch-cu128=https://download.pytorch.org/whl/cu128
uv add tensorboard
uv lock
uv sync
~~~

Inspect the generated file and add `explicit = true` to the index.

## 6. `requirements.txt` format

Install Torch wheels first from their selected index:

~~~text
# requirements-pytorch-cu128.txt
--index-url https://download.pytorch.org/whl/cu128
torch==2.11.0
torchvision==0.26.0
torchaudio==2.11.0
~~~

Keep the project requirements separately:

~~~text
# requirements.txt
nemo-toolkit[asr]>=2.3.0
transformers>=4.57.3
tensorboard>=2.20.0
pydub>=0.25.1
~~~

~~~powershell
python -m pip install -r requirements-pytorch-cu128.txt
python -m pip install -r requirements.txt
~~~

A one-file alternative puts `--extra-index-url https://download.pytorch.org/whl/cu128` first, followed by every dependency. Prefer `pyproject.toml` plus `uv.lock`: it records the resolved versions and package source mapping.

## 7. Final PyTorch validation cell

The correct spelling is `torch.cuda.is_available()`, not `torch.is_cuda_avaialable`.

~~~python
import torch
print("PyTorch version:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA unavailable: check driver and CUDA PyTorch wheel."
device = torch.device("cuda:0")
print("GPU:", torch.cuda.get_device_name(device))
print("Device capability:", torch.cuda.get_device_capability(device))
x = torch.randn((4096, 4096), device=device)
y = x @ x
torch.cuda.synchronize()
print("Matrix multiply succeeded:", y.shape, y.device)
print("Allocated GB:", round(torch.cuda.memory_allocated(device) / 2**30, 3))
~~~

- [ ] `torch.cuda.is_available()` is `True`.
- [ ] `torch.version.cuda` is not `None`.
- [ ] The intended GPU name appears.
- [ ] The matrix multiplication succeeds.

## 8. Optional TensorBoard check

~~~python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/gpu_preflight")
writer.add_scalar("preflight/cuda_available", int(torch.cuda.is_available()), 0)
writer.close()
~~~

~~~powershell
uv run tensorboard --logdir runs
~~~

## 9. If CUDA validation fails

1. Re-run `nvidia-smi`; repair/update the NVIDIA driver if it fails.
2. Run `uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"`. `None` means a CPU build is installed.
3. Ensure `torch`, `torchvision`, and `torchaudio` come from one selected CUDA index.
4. Recreate the virtual environment from the corrected lockfile rather than mixing manual installs.
5. Install the standalone Toolkit only when compiling CUDA code; it is not the usual repair for a wrong PyTorch wheel.

## References

- [PyTorch Start Locally selector](https://pytorch.org/get-started/locally/)
- [Official PyTorch CUDA wheel/version matrix](https://pytorch.org/get-started/previous-versions/)
- [uv with PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [uv package indexes](https://docs.astral.sh/uv/concepts/indexes/)
- [NVIDIA CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)
