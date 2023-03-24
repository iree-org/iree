## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utils that help construct definitions."""

from typing import Callable, List, Sequence

MAX_SUBSTITUTION_ITERATIONS = 10


def materialize_flags(flags: Sequence[str],
                      map_funcs: Sequence[Callable[[str], str]]) -> List[str]:
  """Call map functions to materialize flag values.

  It parses and extracts the flag value from both keyword and positional flags.
  Each flag value is proccessed by the map functions until reaching the fixed
  point, then replaced the original flag value.
  
  Args:
    flags: list of flags.
    map_funcs: list of map functions to map flag value.
  Returns:
    List of materialized flags.
  """

  materialized_flags = []
  for flag in flags:
    if flag.startswith("-"):
      # Keyward argument
      value_pos = flag.find("=") + 1
      # Do nothing if there is no flag value.
      if value_pos == 0:
        materialized_flags.append(flag)
        continue
    else:
      # Positional argument
      value_pos = 0

    prev_value = flag[value_pos:]
    # Iteratively replace until reaching the fixed point.
    new_value = prev_value
    iterations = 0
    while True:
      for map_func in map_funcs:
        new_value = map_func(new_value)
      if new_value == prev_value:
        break
      iterations += 1
      if iterations > MAX_SUBSTITUTION_ITERATIONS:
        raise ValueError(f"Too many iterations to materialize: {flag}")
      prev_value = new_value

    materialized_flags.append(flag[:value_pos] + new_value)

  return materialized_flags
