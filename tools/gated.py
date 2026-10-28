"""Run one command once fewer than 10 study compute processes are busy (the study's shared-machine rule; see make_references.py).

usage: python tools/gated.py COMMAND [ARGS ...]   (exit code of the command)
"""
import subprocess
import sys
import time

from make_references import busy

while busy() >= 10:
    time.sleep(2)
print(time.strftime("%H:%M:%S"), "start", " ".join(sys.argv[1:]), flush=True)
sys.exit(subprocess.call(sys.argv[1:]))
