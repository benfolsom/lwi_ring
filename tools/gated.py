"""Optional compute gate: start a command only while fewer than N matching processes are busy.

usage: python tools/gated.py --busy-pattern=REGEX [--max-busy=10] COMMAND [ARGS ...]   (exit code of the command)

A process counts as busy when it is a python process at >= 25% CPU whose command line matches REGEX (case-insensitive).
"""
import re
import subprocess
import sys
import time


def busy(pattern: str) -> int:
    out = subprocess.run(["ps", "-axo", "%cpu=,command="], capture_output=True, text=True).stdout
    n = 0
    for line in out.splitlines():
        cpu, _, cmd = line.strip().partition(" ")
        try:
            hot = float(cpu) >= 25
        except ValueError:
            continue
        n += hot and "python" in cmd.split(" ")[0].lower() and bool(re.search(pattern, cmd, re.I))
    return n


if __name__ == "__main__":
    opts = {}
    args = sys.argv[1:]
    while args and args[0].startswith("--"):
        key, _, value = args.pop(0).partition("=")
        opts[key] = value
    if "--busy-pattern" not in opts or not args:
        sys.exit(__doc__)
    limit = int(opts.get("--max-busy", "10"))
    while busy(opts["--busy-pattern"]) >= limit:
        time.sleep(2)
    print(time.strftime("%H:%M:%S"), "start", " ".join(args), flush=True)
    sys.exit(subprocess.call(args))
