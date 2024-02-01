# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
    from iree.compiler.extras import fx_importer
except ModuleNotFoundError as e:
    while e is not None:
        if isinstance(e, ModuleNotFoundError) and e.name == "torch":
            break
        e = e.__cause__
    else:
        raise ModuleNotFoundError(
            "Failed to import the fx_importer (for a reason other than torch "
            "not being found)"
        ) from e
