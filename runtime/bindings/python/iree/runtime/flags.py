# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._binding import parse_flags

# When enabled, performs additional function input validation checks. In the
# event of errors, this will yield nicer error messages but comes with a
# runtime cost.
FUNCTION_INPUT_VALIDATION = True
