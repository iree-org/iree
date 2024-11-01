# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.build import *


@entrypoint(description="Compiles an mnist model")
def mnist(
    url=cl_arg(
        "mnist-onnx-url",
        default="https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx",
        help="URL from which to download mnist",
    ),
):
    fetch_http(
        name="mnist.onnx",
        url=url,
    )
    onnx_import(
        name="mnist.mlir",
        source="mnist.onnx",
    )
    return compile(
        name="mnist",
        source="mnist.mlir",
    )


if __name__ == "__main__":
    iree_build_main()
