# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import threading
from collections.abc import Sequence
from contextlib import contextmanager

from iree.compiler.ir import Value

_thread_state = threading.local()

###############################################################################
# Tracing intrinsics
###############################################################################


class ProcedureTraceError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class IrTrace:
    """Gets callbacks for tracing events."""

    __slots__ = []

    def finalize(self):
        """Called when the trace is finished (popped off the stack)."""
        pass

    def handle_call(self, target: "Intrinsic", args, kwargs):
        raise NotImplementedError("The current trace scope does not support calls")

    def handle_assignment(self, scope, target, updated_value):
        raise NotImplementedError("The current trace scope does not support assignment")


def _trace_scopes() -> list[IrTrace]:
    try:
        trace_scopes = _thread_state.trace_scopes
    except AttributeError:
        trace_scopes = _thread_state.trace_scopes = []
    return trace_scopes


@contextmanager
def new_ir_trace_scope(ir_trace: IrTrace):
    trace_scopes = _trace_scopes()
    trace_scopes.append(ir_trace)
    try:
        yield ir_trace
    finally:
        ir_trace.finalize()
        del trace_scopes[-1]


def current_ir_trace() -> IrTrace:
    return _trace_scopes()[-1]


class Intrinsic:
    """Objects which interact natively with the tracing system implement this."""

    __slots__: list[str] = []

    def resolve_ir_values(self, proc_trace: "IrTrace") -> Sequence[Value]:
        raise NotImplementedError(
            f"Cannot use {self} as an expression in a procedural function"
        )

    # Helpers for accessing the ir_value within the current trace.
    @property
    def ir_values(self) -> Sequence[Value]:
        return self.resolve_ir_values(current_ir_trace())

    @property
    def ir_value(self) -> Value:
        values = self.ir_values
        assert len(values) == 1, "Expected arity one intrinsic"
        return values[0]


class CallableIntrinsic(Intrinsic):
    """Intrinsic subclass that supports calls.
    This is separate so as to make error handling better (i.e. does not support
    calls) for intrinsics that are not callable.
    """

    __slots__ = []

    def __call__(self, *args, **kwargs):
        return current_ir_trace().handle_call(self, args, kwargs)
