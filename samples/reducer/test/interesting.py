#!/usr/bin/python3

import sys
import subprocess

out = subprocess.run(
    [
        "../../../build/tools/iree-compile",
        sys.argv[1],
        "--iree-hal-target-backends=vmvx",
        "--compile-to=executable-targets",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    universal_newlines=True,
)

if out.returncode != 0:
    exit(1)

if "vm.add" in out.stdout:
    exit(0)
else:
    exit(1)
