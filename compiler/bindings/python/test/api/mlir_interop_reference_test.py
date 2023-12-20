# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# The iree compiler API includes facilities for importing/exporting
# contexts and modules from the MLIR Python bindings. This is a tricky
# interchange and we test some explicit access patterns for issues.

import gc

from iree.compiler.api import (
    Invocation,
    Session,
    Source,
    Output,
)

from iree.compiler import ir


def test_context_interchange():
    print("*** test_context_interchange")
    s = Session()
    start_live_contexts = ir.Context._get_live_count()
    print("LIVE PY CONTEXTS:", start_live_contexts)
    print("GETTING PYTHON CONTEXT")
    context = s.context

    print("DELETING SESSION")
    s = None
    gc.collect()

    print("LIVE PY CONTEXTS:", ir.Context._get_live_count())
    context = None
    gc.collect()
    end_live_contexts = ir.Context._get_live_count()
    print("LIVE PY CONTEXTS:", end_live_contexts)
    assert start_live_contexts == end_live_contexts


def test_import_module():
    print("*** test_import_module")
    s = Session()
    context = s.context
    with ir.Location.unknown(context):
        module_op = ir.Module.create().operation
        inv = s.invocation()
        inv.import_module(module_op)

    gc.collect()
    module_op = None
    gc.collect()

    session = None
    gc.collect()

    inv = None
    gc.collect()
    print("All done:", context._get_live_module_count())


def test_export_module():
    print("*** test_export_module")
    s = Session()
    start_live_operations = 0
    # Make it known to Python.
    s.context
    source = Source.wrap_buffer(s, b"builtin.module {}")
    inv = s.invocation()
    assert inv.parse_source(source)

    module_op = inv.export_module()
    print(module_op)

    context = None
    gc.collect()
    print("Exported live operations:", module_op.context._get_live_operation_count())

    inv = None
    gc.collect()
    print("Exported live operations:", module_op.context._get_live_operation_count())

    source = None
    gc.collect()
    print("Exported live operations:", module_op.context._get_live_operation_count())

    s = None
    gc.collect()
    print("Exported live operations:", module_op.context._get_live_operation_count())

    context = module_op.context
    module_op = None
    gc.collect()
    print("Exported live operations:", context._get_live_operation_count())
    assert start_live_operations == context._get_live_operation_count()

    context = None
    gc.collect()


test_context_interchange()
test_import_module()
test_export_module()
