## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utils that help construct definitions."""

import string
from typing import Any, Callable, List, Sequence

MAX_SUBSTITUTION_ITERATIONS = 10


def transform_flags(flags: Sequence[str],
                    map_funcs: Sequence[Callable[[str], str]]) -> List[str]:
  """Call map functions to transform flag values, e.g., replace placeholders
  that were unknown when the flag was constructed.

  It parses and extracts the flag values from both keyword and positional flags,
  transforms them, and returns the updated flags with transformed values.

  Each flag value is transformed only once by each map function in order.
  
  Args:
    flags: list of flags.
    map_funcs: list of map functions to map flag value.
  Returns:
    List of transformed flags.
  """

  transformed_flags = []
  for flag in flags:
    keyword, separator, value = ("", "", flag)
    if flag.startswith("-"):
      keyword, separator, value = flag.partition("=")

    if value:
      for map_func in map_funcs:
        value = map_func(value)

    transformed_flags.append(f"{keyword}{separator}{value}")

  return transformed_flags


def substitute_flag_vars(flags: Sequence[str], **mapping: Any) -> List[str]:
  """Sugar of transform_flags to substitute variables in string.Template format.
  """
  return transform_flags(
      flags=flags,
      map_funcs=[lambda value: string.Template(value).substitute(mapping)])
