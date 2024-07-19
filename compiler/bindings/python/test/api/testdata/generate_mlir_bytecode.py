# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.compiler.api import (
    Session,
    Source,
    Output,
)
import os
import iree.compiler.tools.tflite


def generate_test_bytecode():
    session = Session()
    inv = session.invocation()
    source = Source.wrap_buffer(session, b"builtin.module {}")
    inv.parse_source(source)
    out = Output.open_membuffer()
    inv.output_ir_bytecode(out)
    mem = out.map_memory()

    this_dir = os.path.dirname(__file__)
    with open(os.path.join(this_dir, "bytecode_testfile.bc"), "wb") as file:
        file.write(bytes(mem))


def generate_zero_terminated_bytecode():
    """MLIR Bytecode can also be zero terminated. I couldn't find a way to generate zero terminated
    bytecode apart from this. Printing as textual IR and then reparsing and printing as bytecode
    removes the zero termination on this IR. This might very well be an odity of TF."""
    if not iree.compiler.tools.tflite.is_available():
        return
    this_dir = os.path.dirname(__file__)
    path = os.path.join(this_dir, "..", "..", "tools", "testdata", "tflite_sample.fb")
    bytecode = iree.compiler.tools.tflite.compile_file(path, import_only=True)
    with open(
        os.path.join(this_dir, "bytecode_zero_terminated_testfile.bc"), "wb"
    ) as file:
        file.write(bytecode)


if __name__ == "__main__":
    generate_test_bytecode()
    generate_zero_terminated_bytecode()
