# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


def write_benchmark_result(result: list[str], save_path: str):
  """Writes an array to file as a comma-separated line."""
  results_array = [str(i) for i in result]
  print("Writing " + str(results_array))
  with open(save_path, "a") as f:
    f.write(",".join(results_array) + "\n")
