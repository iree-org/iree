# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools as _functools

from typing import Sequence
from .base import RtlBuilder, RtlModule


def _get_std_rtl_modules() -> Sequence[RtlModule]:
  from .modules import (
      booleans,
      numerics,
  )
  return [m.RTL_MODULE for m in (booleans, numerics)]


STD_RTL_MODULES = _get_std_rtl_modules()

# Source bundle for the standard RTL.
get_std_rtl_source_bundle = RtlBuilder.lazy_build_source_bundle(STD_RTL_MODULES)
