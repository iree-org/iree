# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
    from iree.compiler.extras import fx_importer

    print("fx_importer imported successfully")
except ModuleNotFoundError as e:
    e_orig = e
    while e is not None:
        if isinstance(e, ModuleNotFoundError) and e.name == "torch":
            print("torch not found, skipping fx_importer_test")
            break
        e = e.__cause__
    else:
        raise ModuleNotFoundError(
            f"Failed to import the fx_importer (for a reason other than torch not being found)"
        ) from e_orig
