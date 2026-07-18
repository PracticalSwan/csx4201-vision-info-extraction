"""D:-backed runtime environment setup and compatibility reporting."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.rotation_common import atomic_write_json

DEFAULT_ASSET_ROOT = Path("D:/CSX4201/vision-info-extraction-assets")
_DLL_HANDLES: list[Any] = []


def configure_external_environment(
    asset_root: str | Path | None = None,
) -> dict[str, str]:
    """Configure caches, honoring the portable runtime override when omitted."""
    root = Path(
        asset_root
        or os.environ.get("OCR_MODEL_ASSET_ROOT")
        or DEFAULT_ASSET_ROOT
    )
    values = {
        "PADDLE_PDX_CACHE_HOME": str(root / "cache" / "paddlex"),
        "HF_HOME": str(root / "cache" / "huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(root / "cache" / "huggingface" / "hub"),
        "TRANSFORMERS_CACHE": str(root / "cache" / "huggingface" / "transformers"),
        "TORCH_HOME": str(root / "cache" / "torch"),
        "PIP_CACHE_DIR": str(root / "cache" / "pip"),
        "TMP": str(root / "cache" / "temp"),
        "TEMP": str(root / "cache" / "temp"),
    }
    for path in values.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    for name, value in values.items():
        os.environ[name] = value
    configure_windows_nvidia_dlls()
    return values


def configure_windows_nvidia_dlls(prefix: str | Path | None = None) -> list[str]:
    """Expose CUDA DLLs bundled by Paddle's NVIDIA wheel dependencies."""
    if os.name != "nt":
        return []
    environment = Path(prefix or sys.prefix)
    nvidia_root = environment / "Lib" / "site-packages" / "nvidia"
    candidates = [
        nvidia_root / "cu13" / "bin" / "x86_64",
        nvidia_root / "cudnn" / "bin",
    ]
    existing = [str(path) for path in candidates if path.is_dir()]
    if not existing:
        return []
    os.environ["PATH"] = os.pathsep.join(existing + [os.environ.get("PATH", "")])
    if hasattr(os, "add_dll_directory"):
        for path in existing:
            try:
                _DLL_HANDLES.append(os.add_dll_directory(path))
            except OSError:
                pass
    return existing


def collect_environment_report(
    asset_root: str | Path = DEFAULT_ASSET_ROOT,
    layout_python: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(asset_root)
    environment_values = configure_external_environment(root)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "asset_root": str(root),
        "environment": environment_values,
        "drives": {},
        "packages": {},
        "gpu": _nvidia_report(),
    }
    for drive in (Path("C:/"), Path("D:/")):
        if drive.exists():
            usage = shutil.disk_usage(drive)
            report["drives"][str(drive)] = {
                "total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free,
                "free_gib": round(usage.free / 1024**3, 3),
            }
    for package in ("paddlepaddle-gpu", "paddlepaddle", "paddleocr", "paddlex", "torch", "transformers"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    # Paddle and PyTorch bundle incompatible cuDNN DLL builds on this Windows
    # host. Probe them in isolated processes, exactly as end-to-end inference
    # isolates OCR from the layout model.
    layout_executable = Path(layout_python) if layout_python else (
        root / "environments" / "ie-layout" / "Scripts" / "python.exe"
    )
    report["paddle_runtime"] = _framework_runtime_probe("paddle", sys.executable)
    report["torch_runtime"] = _framework_runtime_probe("torch", layout_executable)
    report["layout_python_executable"] = str(layout_executable)
    report["layout_packages"] = _package_probe(layout_executable)
    report["framework_process_isolation_required"] = os.name == "nt"
    report["storage_gate"] = storage_gate(root)
    return report


def storage_gate(
    asset_root: str | Path = DEFAULT_ASSET_ROOT,
    *,
    minimum_c_free_gib: float = 15.0,
    minimum_asset_free_gib: float = 15.0,
    anticipated_c_gib: float = 0.0,
    anticipated_asset_gib: float = 0.0,
) -> dict[str, Any]:
    """Report whether anticipated model work preserves both drive reserves."""
    root = Path(asset_root)
    c_free_gib: float | None = None
    asset_free_gib: float | None = None
    errors: list[str] = []
    if Path("C:/").exists():
        c_free_gib = shutil.disk_usage("C:/").free / 1024**3
    if not root.drive or not Path(root.anchor).exists():
        errors.append(f"external asset drive is unavailable: {root}")
    else:
        asset_free_gib = shutil.disk_usage(root.anchor).free / 1024**3
    result = evaluate_storage_reserve(
        c_free_gib=c_free_gib,
        asset_free_gib=asset_free_gib,
        minimum_c_free_gib=minimum_c_free_gib,
        minimum_asset_free_gib=minimum_asset_free_gib,
        anticipated_c_gib=anticipated_c_gib,
        anticipated_asset_gib=anticipated_asset_gib,
    )
    result["errors"] = errors + result["errors"]
    result["passed"] = not result["errors"]
    return result


def evaluate_storage_reserve(
    *,
    c_free_gib: float | None,
    asset_free_gib: float | None,
    minimum_c_free_gib: float = 15.0,
    minimum_asset_free_gib: float = 15.0,
    anticipated_c_gib: float = 0.0,
    anticipated_asset_gib: float = 0.0,
) -> dict[str, Any]:
    """Pure reserve calculation used by the live gate and unit tests."""
    if min(
        minimum_c_free_gib,
        minimum_asset_free_gib,
        anticipated_c_gib,
        anticipated_asset_gib,
    ) < 0:
        raise ValueError("storage reserve and anticipated-write values must be non-negative")
    result: dict[str, Any] = {
        "minimum_c_free_gib": minimum_c_free_gib,
        "minimum_asset_free_gib": minimum_asset_free_gib,
        "anticipated_c_gib": anticipated_c_gib,
        "anticipated_asset_gib": anticipated_asset_gib,
        "passed": True,
        "errors": [],
    }
    if c_free_gib is not None:
        projected = c_free_gib - anticipated_c_gib
        result.update({
            "c_free_gib": round(c_free_gib, 3),
            "projected_c_free_gib": round(projected, 3),
        })
        if projected < minimum_c_free_gib:
            result["errors"].append(
                f"C: projected free space {projected:.2f} GiB is below the "
                f"{minimum_c_free_gib:.2f} GiB reserve"
            )
    if asset_free_gib is not None:
        projected = asset_free_gib - anticipated_asset_gib
        result.update({
            "asset_free_gib": round(asset_free_gib, 3),
            "projected_asset_free_gib": round(projected, 3),
        })
        if projected < minimum_asset_free_gib:
            result["errors"].append(
                f"asset drive projected free space {projected:.2f} GiB is below the "
                f"{minimum_asset_free_gib:.2f} GiB reserve"
            )
    result["passed"] = not result["errors"]
    return result


def require_storage_gate(
    asset_root: str | Path = DEFAULT_ASSET_ROOT,
    *,
    operation: str,
    anticipated_c_gib: float = 0.0,
    anticipated_asset_gib: float = 0.0,
) -> dict[str, Any]:
    """Abort a large operation before it can cross a configured reserve."""
    result = storage_gate(
        asset_root,
        anticipated_c_gib=anticipated_c_gib,
        anticipated_asset_gib=anticipated_asset_gib,
    )
    if not result["passed"]:
        raise RuntimeError(f"storage gate blocked {operation}: {'; '.join(result['errors'])}")
    return result


def write_environment_report(
    output: str | Path,
    asset_root: str | Path = DEFAULT_ASSET_ROOT,
    layout_python: str | Path | None = None,
) -> dict[str, Any]:
    report = collect_environment_report(asset_root, layout_python)
    atomic_write_json(Path(output), report)
    return report


def _nvidia_report() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False}
    command = [
        executable,
        "--query-gpu=name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=15)
        values = [part.strip() for part in completed.stdout.strip().split(",")]
        return {
            "available": True,
            "name": values[0] if len(values) > 0 else None,
            "driver_version": values[1] if len(values) > 1 else None,
            "memory_mib": int(values[2]) if len(values) > 2 and values[2].isdigit() else None,
            "compute_capability": values[3] if len(values) > 3 else None,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": True, "query_error": f"{type(exc).__name__}: {exc}"}


def _framework_runtime_probe(
    framework: str, python_executable: str | Path | None = None
) -> dict[str, Any]:
    programs = {
        "paddle": (
            "import json,paddle; print(json.dumps({"
            "'compiled_with_cuda':bool(paddle.device.is_compiled_with_cuda()),"
            "'device':paddle.device.get_device(),"
            "'cuda_device_count':int(paddle.device.cuda.device_count())}))"
        ),
        "torch": (
            "import json,torch; available=bool(torch.cuda.is_available()); print(json.dumps({"
            "'cuda_available':available,'cuda_version':torch.version.cuda,"
            "'device_count':int(torch.cuda.device_count()),"
            "'device_name':torch.cuda.get_device_name(0) if available else None,"
            "'device_capability':list(torch.cuda.get_device_capability(0)) if available else None}))"
        ),
    }
    if framework not in programs:
        raise ValueError(framework)
    try:
        executable = Path(python_executable or sys.executable)
        if not executable.is_file():
            return {"error": f"Python executable is missing: {executable}"}
        completed = subprocess.run(
            [str(executable), "-c", programs[framework]],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
            check=True,
        )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        return json.loads(lines[-1])
    except Exception as exc:
        details = ""
        if isinstance(exc, subprocess.CalledProcessError):
            details = (exc.stderr or exc.stdout or "")[-2000:].strip()
        return {"error": f"{type(exc).__name__}: {details or exc}"}


def _package_probe(python_executable: str | Path) -> dict[str, Any]:
    executable = Path(python_executable)
    if not executable.is_file():
        return {"error": f"Python executable is missing: {executable}"}
    program = (
        "import importlib.metadata,json; names=['torch','transformers','sentencepiece','accelerate']; "
        "print(json.dumps({name:(importlib.metadata.version(name) if any(d.metadata['Name'].lower()==name "
        "for d in importlib.metadata.distributions()) else None) for name in names}))"
    )
    try:
        completed = subprocess.run(
            [str(executable), "-c", program], capture_output=True, text=True,
            encoding="utf-8", timeout=60, check=True,
        )
        return json.loads(completed.stdout.splitlines()[-1])
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
