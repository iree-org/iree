## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Helpers for path generation."""

import re

# Match any character that are not friendly in filesystem path.
DISALLOWED_TARGET_CHAR_MATCHER = re.compile(r"[^A-Za-z0-9_.+-]")


def sanitize_path_name(name: str) -> str:
  return DISALLOWED_TARGET_CHAR_MATCHER.sub("_", name)
