from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def resolve_from_root(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def venv_python() -> Path:
    if os.name == "nt":
        candidate = ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = ROOT / ".venv" / "bin" / "python"

    if candidate.exists():
        return candidate.resolve()

    # The launcher itself uses only stdlib, but the workers need the project venv.
    # Falling back to the current interpreter is useful when it already is the venv.
    return Path(sys.executable).resolve()


def http_json(url: str, timeout: float = 1.0) -> dict[str, Any] | None:
    req = urllib.request.Request(
        url,
        headers={"Authorization": "Bearer local-no-key"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", errors="replace")
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        json.JSONDecodeError,
        OSError,
    ):
        return None


def models_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/models"


def stream_output(proc: subprocess.Popen, prefix: str) -> None:
    if proc.stdout is None:
        return

    try:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            print(f"{prefix} {line}", end="", flush=True)
    except Exception:
        pass


def start_logged_process(
    cmd: list[str],
    *,
    cwd: Path,
    prefix: str,
    env: dict[str, str] | None = None,
) -> subprocess.Popen:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    threading.Thread(
        target=stream_output,
        args=(proc, prefix),
        daemon=True,
    ).start()
    return proc


def terminate_process(proc: subprocess.Popen | None, name: str) -> None:
    if proc is None or proc.poll() is not None:
        return

    print(f"[launcher] stopping {name}...")
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)
    except OSError:
        pass


def prepare_harness_config(
    *,
    harness_dir: Path,
    source_name: str,
    local_server_cfg: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    source = harness_dir / source_name
    cfg = load_json(source)

    # harness.py resolves capture_root from its own CWD.
    # Since the Harness lives under <root>/vlm_harness_...,
    # the real capture directory is one level above.
    cfg["capture_root"] = "../capture"

    # When the launcher owns a local llama.cpp server, make sure the generated
    # worker config points at the exact same host/port/model alias.
    if local_server_cfg.get("enabled", True):
        provider = cfg.setdefault("provider", {})
        host = str(local_server_cfg.get("host", "127.0.0.1"))
        port = int(local_server_cfg.get("port", 8080))
        provider["base_url"] = f"http://{host}:{port}/v1"
        provider["model"] = str(
            local_server_cfg.get("alias", provider.get("model", "qwen2.5-vl-3b"))
        )

    target = harness_dir / "harness_config.integrated.json"
    write_json(target, cfg)
    return target, cfg


def preflight(
    *,
    py: Path,
    vision_script: Path,
    tracker: Path,
    model: Path,
    harness_dir: Path,
    harness_cfg_source: Path,
    server_cfg: dict[str, Any],
) -> None:
    required = {
        "Python": py,
        "Vision UI": vision_script,
        "BoT-SORT config": tracker,
        "YOLOE model": model,
        "Harness": harness_dir / "harness.py",
        "Provider": harness_dir / "provider.py",
        "VLM schema": harness_dir / "vision_response.schema.json",
        "Harness config": harness_cfg_source,
    }

    missing = [
        f"{name}: {path}"
        for name, path in required.items()
        if not path.exists()
    ]

    if server_cfg.get("enabled", True):
        server_exe = Path(str(server_cfg["exe"]))
        if not server_exe.exists():
            missing.append(f"llama-server.exe: {server_exe}")

    if missing:
        print("[launcher] preflight failed:")
        for item in missing:
            print("  -", item)
        raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Integrated launcher: llama-server -> VLM Harness -> "
            "YOLOE camera/tracking/HUD"
        )
    )
    parser.add_argument(
        "--config",
        default="integrated_config.json",
        help="Launcher configuration relative to the project root.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Override camera index from integrated_config.json.",
    )
    parser.add_argument(
        "--keep-server",
        action="store_true",
        help="Do not stop llama-server when the camera UI exits.",
    )
    parser.add_argument(
        "vision_args",
        nargs=argparse.REMAINDER,
        help=(
            "Extra args forwarded to yoloe_desktop_pipeline_v2_2_ui.py. "
            "Put them after --, e.g. -- --conf 0.30"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config_path = resolve_from_root(args.config)
    cfg = load_json(config_path)

    py = venv_python()

    vision_script = resolve_from_root(
        cfg.get("vision_script", "yoloe_desktop_pipeline_v2_2_ui.py")
    )
    tracker = resolve_from_root(
        cfg.get("tracker", "botsort_yoloe_desktop.yaml")
    )
    model = resolve_from_root(
        cfg.get("model", "yoloe-26s-seg-pf.pt")
    )
    harness_dir = resolve_from_root(
        cfg.get("harness_dir", "vlm_harness_qwen25vl_v1")
    )
    harness_cfg_name = str(
        cfg.get("harness_config", "harness_config.json")
    )
    harness_cfg_source = harness_dir / harness_cfg_name
    server_cfg = dict(cfg.get("local_vlm_server", {}))

    preflight(
        py=py,
        vision_script=vision_script,
        tracker=tracker,
        model=model,
        harness_dir=harness_dir,
        harness_cfg_source=harness_cfg_source,
        server_cfg=server_cfg,
    )

    integrated_harness_cfg_path, harness_cfg = prepare_harness_config(
        harness_dir=harness_dir,
        source_name=harness_cfg_name,
        local_server_cfg=server_cfg,
    )

    provider_base_url = str(
        harness_cfg["provider"]["base_url"]
    ).rstrip("/")
    health_url = models_url(provider_base_url)

    camera = (
        args.camera
        if args.camera is not None
        else int(cfg.get("camera", 0))
    )

    print("=" * 68)
    print(" YOLOE + VLM + HUD Integrated Launcher")
    print("=" * 68)
    print(f"[launcher] root       : {ROOT}")
    print(f"[launcher] python     : {py}")
    print(f"[launcher] vision     : {vision_script.name}")
    print(f"[launcher] camera     : {camera}")
    print(f"[launcher] harness    : {harness_dir.name}")
    print(f"[launcher] VLM API    : {provider_base_url}")
    print()

    llama_proc: subprocess.Popen | None = None
    harness_proc: subprocess.Popen | None = None

    try:
        # 1) Start/reuse local VLM server.
        health = http_json(health_url, timeout=1.0)

        if health is not None:
            print("[launcher] VLM server already healthy; reusing it.")
        elif server_cfg.get("enabled", True):
            server_exe = Path(str(server_cfg["exe"])).resolve()
            repo = str(
                server_cfg.get(
                    "repo",
                    "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
                )
            )
            alias = str(server_cfg.get("alias", "qwen2.5-vl-3b"))
            host = str(server_cfg.get("host", "127.0.0.1"))
            port = int(server_cfg.get("port", 8080))
            gpu_layers = int(server_cfg.get("gpu_layers", 99))
            context = int(server_cfg.get("context", 4096))

            server_cmd = [
                str(server_exe),
                "-hf",
                repo,
                "--alias",
                alias,
                "--host",
                host,
                "--port",
                str(port),
                "-ngl",
                str(gpu_layers),
                "-c",
                str(context),
            ]

            extra_server_args = server_cfg.get("extra_args", [])
            if isinstance(extra_server_args, list):
                server_cmd.extend(str(x) for x in extra_server_args)

            print("[launcher] starting local VLM server...")
            llama_proc = start_logged_process(
                server_cmd,
                cwd=server_exe.parent,
                prefix="[llama]",
            )

            deadline = time.monotonic() + float(
                server_cfg.get("startup_timeout_s", 90)
            )

            while time.monotonic() < deadline:
                if llama_proc.poll() is not None:
                    raise RuntimeError(
                        "llama-server exited before becoming healthy."
                    )

                health = http_json(health_url, timeout=1.0)
                if health is not None:
                    break

                time.sleep(0.5)

            if health is None:
                raise TimeoutError(
                    f"VLM server did not become healthy: {health_url}"
                )

            print("[launcher] local VLM server ready.")
        else:
            raise RuntimeError(
                "VLM API is not reachable and local_vlm_server.enabled=false."
            )

        # 2) Start Harness worker.
        print("[launcher] starting VLM Harness worker...")

        worker_env = os.environ.copy()
        worker_env["PYTHONUNBUFFERED"] = "1"

        harness_proc = start_logged_process(
            [
                str(py),
                "-u",
                "harness.py",
                "--config",
                integrated_harness_cfg_path.name,
            ],
            cwd=harness_dir,
            prefix="[harness]",
            env=worker_env,
        )

        time.sleep(0.8)

        if harness_proc.poll() is not None:
            raise RuntimeError(
                "Harness worker exited immediately. "
                "Check the [harness] log above."
            )

        print("[launcher] Harness worker ready.")

        # 3) Start camera / YOLOE / HUD in foreground.
        vision_cmd = [
            str(py),
            str(vision_script),
            "--camera",
            str(camera),
            "--model",
            str(model),
            "--tracker",
            str(tracker),
            "--output",
            str((ROOT / "capture").resolve()),
        ]

        forwarded = list(args.vision_args)
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        vision_cmd.extend(forwarded)

        print()
        print("[launcher] starting camera / YOLOE / HUD...")
        print("[launcher] flow:")
        print("  camera -> YOLOE -> select -> capture/queue")
        print("         -> Harness -> VLM -> validated record.json -> HUD")
        print()
        print("[launcher] close the camera window or press Q/Esc to stop.")
        print("-" * 68)

        vision_proc = subprocess.Popen(
            vision_cmd,
            cwd=str(ROOT),
        )

        return_code = vision_proc.wait()

        print("-" * 68)
        print(f"[launcher] Vision UI exited with code {return_code}.")
        return int(return_code)

    except KeyboardInterrupt:
        print("\n[launcher] Ctrl+C received.")
        return 130

    except Exception as exc:
        print(f"\n[launcher] ERROR: {type(exc).__name__}: {exc}")
        return 1

    finally:
        terminate_process(harness_proc, "Harness worker")

        if llama_proc is not None and not args.keep_server:
            terminate_process(llama_proc, "llama-server")

        print("[launcher] shutdown complete.")


if __name__ == "__main__":
    raise SystemExit(main())
