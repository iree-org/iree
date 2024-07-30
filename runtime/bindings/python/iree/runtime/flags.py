# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Global runtime flags.

Obligatory disclaimer: there are APIs for managing most of this in a library
friendly way, and those are almost always better. However, flags as are
typically passed to `iree-run-module` can be set *process wide* by using
the `parse_flags` API below.

In addition, the `IREE_PY_RUNTIME_FLAGS` environment variable, if present,
will cause the included flags to be parsed at import time.
"""

from ._binding import parse_flags

# When enabled, performs additional function input validation checks. In the
# event of errors, this will yield nicer error messages but comes with a
# runtime cost.
FUNCTION_INPUT_VALIDATION = True


def _load_default_flags_from_env():
    import os
    import shlex

    flags_env_str = os.getenv("IREE_PY_RUNTIME_FLAGS")
    if not flags_env_str:
        return
    flags_env_list = shlex.split(flags_env_str)
    try:
        parse_flags(*flags_env_list)
    except ValueError as e:
        raise RuntimeError(
            f"Bad flag value in environment variable IREE_PY_RUNTIME_FLAGS="
            f"'{flags_env_str}': {e}"
        ) from e


_load_default_flags_from_env()
