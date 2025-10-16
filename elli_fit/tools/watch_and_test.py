"""
===========================================================
Automatic Rebuild + Test Watcher (Rootdir+Venv Safe)
===========================================================

Fixes common pitfalls:
  - Always runs pip/pytest from the project ROOT (repo root)
  - Shows which Python interpreter is used (venv vs system)
  - Pre-collects tests and prints how many were found
  - Uses --rootdir to make pytest discovery deterministic
  - More robust debouncing and ignores __pycache__/*.pyc

Usage
-----
    # IMPORTANT: run with the SAME Python you use for pytest
    # e.g. after `source .venv/bin/activate`:
    python3 tools/watch_and_test.py
    # or:
    python3 tools/watch_and_test.py --no-install --paths examples

Dependencies
------------
    pip install watchdog pytest colorama
"""

# --- Imports --------------------------------------------------------------

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from colorama import Fore, Style, init as colorama_init


# --- Configuration --------------------------------------------------------

colorama_init(autoreset=True)
ROOT = Path(__file__).resolve().parents[1]         # project root (pyproject.toml)
DEFAULT_SRC_DIRS = [ROOT / "src" / "elli_fit", ROOT / "tests"]
PYTHON = sys.executable


# --- CLI Parsing ----------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="watch_and_test",
        description="Watch files, reinstall (optional), and run pytest on change.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--no-install", action="store_true",
                   help="Do not run 'pip install -e .' before tests.")
    p.add_argument("--paths", nargs="*", default=[],
                   help="Additional paths to watch (folders or files).")
    p.add_argument("--interval", type=float, default=0.5,
                   help="Cooldown between detections (seconds).")
    return p.parse_args(argv)


# --- Utils ----------------------------------------------------------------

def run_cmd(cmd: list[str], capture: bool = False) -> tuple[int, str]:
    """
    Run a command with cwd forced to project ROOT.
    Returns (exit_code, stdout_str_if_captured).
    """
    print(f"{Fore.BLUE}➜ {Style.BRIGHT}{' '.join(cmd)}{Style.RESET_ALL}  {Fore.LIGHTBLACK_EX}(cwd={ROOT}){Style.RESET_ALL}")
    try:
        if capture:
            res = subprocess.run(cmd, cwd=str(ROOT), text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            print(res.stdout)
            return res.returncode, res.stdout
        else:
            proc = subprocess.Popen(cmd, cwd=str(ROOT))
            proc.wait()
            return proc.returncode, ""
    except KeyboardInterrupt:
        return 130, ""


def _print_env_banner():
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Style.BRIGHT}Env check")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"Python: {Fore.LIGHTBLACK_EX}{PYTHON}{Style.RESET_ALL}")
    # Show pytest version with the same interpreter
    run_cmd([PYTHON, "-m", "pytest", "--version"])
    # Warn if not using .venv
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        print(f"VIRTUAL_ENV: {Fore.LIGHTBLACK_EX}{venv}{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}⚠️  Not running inside a virtualenv. "
              f"Make sure this Python matches the one you use manually.{Style.RESET_ALL}")


# --- Test Runner ----------------------------------------------------------

def run_tests(skip_install: bool = False) -> None:
    """
    Reinstall (optional), pre-collect tests (with --rootdir),
    then run full pytest with verbose report.
    """
    _print_env_banner()

    if not skip_install:
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Style.BRIGHT}📦 pip install -e .")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        code1, _ = run_cmd([PYTHON, "-m", "pip", "install", "-e", "."])
        if code1 != 0:
            print(f"{Fore.RED}❌ pip install failed.{Style.RESET_ALL}")
            return
    else:
        print(f"{Fore.YELLOW}⚙️  Skipping reinstall (--no-install){Style.RESET_ALL}")

    # Pre-collect: show how many tests pytest sees from ROOT
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Style.BRIGHT}🔎 Pytest collect (dry-run)")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    collect_cmd = [PYTHON, "-m", "pytest", "--collect-only", "-q", f"--rootdir={ROOT}"]
    code_c, out_c = run_cmd(collect_cmd, capture=True)
    if ("collected 0 items" in out_c.lower()) or ("no tests ran" in out_c.lower()):
        print(f"{Fore.YELLOW}⚠️  Pytest collected 0 tests from ROOT.\n"
              f"   • Ensure tests/ exists at {ROOT/'tests'}\n"
              f"   • Files names must be 'test_*.py'\n"
              f"   • You can run: pytest -q --rootdir={ROOT}{Style.RESET_ALL}")

    # Full run
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Style.BRIGHT}🧪 Running pytest")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    test_cmd = [
        PYTHON, "-m", "pytest",
        "-v", "-rxXs",
        "--maxfail=5",
        "--durations=5",
        f"--rootdir={ROOT}",
    ]
    run_cmd(test_cmd)


# --- Watcher --------------------------------------------------------------

IGNORED_DIR_NAMES = {"__pycache__", ".git", ".idea", ".pytest_cache", ".mypy_cache"}
IGNORED_FILE_SUFFIX = {".pyc"}

class ChangeHandler(FileSystemEventHandler):
    def __init__(self, skip_install: bool, cooldown: float):
        self._skip_install = skip_install
        self._last = 0.0
        self._cooldown = cooldown

    def _should_ignore(self, path: str) -> bool:
        p = Path(path)
        if p.suffix in IGNORED_FILE_SUFFIX:
            return True
        parts = set(p.parts)
        return any(name in parts for name in IGNORED_DIR_NAMES)

    def _trigger(self, path):
        if self._should_ignore(path) or not path.endswith(".py"):
            return
        now = time.time()
        if now - self._last < self._cooldown:
            return
        self._last = now
        print(f"\n{Fore.MAGENTA}🧩 File changed:{Style.RESET_ALL} {path}")
        run_tests(skip_install=self._skip_install)

    def on_modified(self, event):
        if not event.is_directory:
            self._trigger(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self._trigger(event.src_path)


# --- Main -----------------------------------------------------------------

def main(argv=None) -> int:
    args = parse_args(argv)

    watch_dirs = list(DEFAULT_SRC_DIRS)
    for p in args.paths:
        watch_dirs.append(Path(p))

    print(f"{Fore.CYAN}{'='*60}")
    print(f"{Style.BRIGHT}👀 elli_fit Watcher — Rootdir+Venv Safe")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    for d in watch_dirs:
        print(f"  • watching: {Fore.LIGHTBLACK_EX}{d}{Style.RESET_ALL}")
    print(f"  • project root: {Fore.LIGHTBLACK_EX}{ROOT}{Style.RESET_ALL}\n"
          f"Press Ctrl+C to stop.\n")

    handler = ChangeHandler(skip_install=args.no_install, cooldown=args.interval)
    observer = Observer()
    for d in watch_dirs:
        observer.schedule(handler, str(d), recursive=True)

    observer.start()
    try:
        run_tests(skip_install=args.no_install)  # initial run from ROOT
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}🛑 Watcher stopped by user.{Style.RESET_ALL}")
        observer.stop()
    observer.join()
    return 0


# --- Entrypoint -----------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())
