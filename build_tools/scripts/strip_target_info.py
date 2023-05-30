#!/usr/bin/env python3
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Strip LLVM IR of target triple and target-specific attributes
"""

import sys
import re
import os


def main():
  sys.stdout.write(f";\n")
  sys.stdout.write(f"; Processed by {os.path.basename(__file__)}\n")
  sys.stdout.write(f";\n")
  target_triple_regex = re.compile(r'^\s*target triple\s*=\s*"[^"]*"')
  target_cpu_regex = re.compile(r'"target-cpu"="[^"]*"')
  target_features_regex = re.compile(r'"target-features"="[^"]*"')
  tune_cpu_regex = re.compile(r'"tune-cpu"="[^"]*"')

  for line in sys.stdin:
    if "target" in line:
      if re.match(target_triple_regex, line):
        continue
      line = re.sub(target_cpu_regex, '', line)
      line = re.sub(target_features_regex, '', line)
      line = re.sub(tune_cpu_regex, '', line)

    sys.stdout.write(line)


if __name__ == "__main__":
  main()
