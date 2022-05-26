# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
import subprocess
import time

from common.benchmark_command import BenchmarkCommand

# Regexes for retrieving memory information.
_VMHWM_REGEX = re.compile(r".*?VmHWM:.*?(\d+) kB.*")
_VMRSS_REGEX = re.compile(r".*?VmRSS:.*?(\d+) kB.*")
_RSSFILE_REGEX = re.compile(r".*?RssFile:.*?(\d+) kB.*")


def run_command(benchmark_command: BenchmarkCommand) -> list[float]:
  """Runs `benchmark_command` and polls for memory consumption statistics.
  Args:
    benchmark_command: A `BenchmarkCommand` object containing information on how to run the benchmark and parse the output.
  Returns:
    An array containing values for [`latency`, `vmhwm`, `vmrss`, `rssfile`]
  """
  command = benchmark_command.generate_benchmark_command()
  print("\n\nRunning command:\n" + " ".join(command))
  benchmark_process = subprocess.Popen(command,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT)

  # Keep a record of the highest VmHWM corresponding VmRSS and RssFile values.
  vmhwm = 0
  vmrss = 0
  rssfile = 0
  while benchmark_process.poll() is None:
    pid_status = subprocess.run(
        ["cat", "/proc/" + str(benchmark_process.pid) + "/status"],
        capture_output=True)
    output = pid_status.stdout.decode()
    vmhwm_matches = _VMHWM_REGEX.search(output)
    vmrss_matches = _VMRSS_REGEX.search(output)
    rssfile_matches = _RSSFILE_REGEX.search(output)

    if vmhwm_matches and vmrss_matches and rssfile_matches:
      curr_vmhwm = float(vmhwm_matches.group(1))
      if curr_vmhwm > vmhwm:
        vmhwm = curr_vmhwm
        vmrss = float(vmrss_matches.group(1))
        rssfile = float(rssfile_matches.group(1))

    time.sleep(0.5)

  stdout_data, _ = benchmark_process.communicate()

  if benchmark_process.returncode != 0:
    print(f"Warning! Benchmark command failed with return code:"
          f" {benchmark_process.returncode}")
    return [0, 0, 0, 0]
  else:
    print(stdout_data.decode())

  latency_ms = benchmark_command.parse_latency_from_output(stdout_data.decode())
  return [latency_ms, vmhwm, vmrss, rssfile]
