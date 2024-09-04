# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
    from iree.compiler.extras import onnx_importer

    print("onnx_importer imported successfully")
except ModuleNotFoundError as e:
    e_orig = e
    while e is not None:
        if isinstance(e, ModuleNotFoundError) and e.name == "onnx":
            print("onnx not found, skipping onnx_importer_test")
            break
        e = e.__cause__
    else:
        raise ModuleNotFoundError(
            "Failed to import the onnx_importer (for a reason other than onnx not being found)"
        ) from e_orig
