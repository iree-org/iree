# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Exports stateless Torch programs through Torchax to JAX and StableHLO.

Read-only parameters, buffers and lifted constants remain explicit inputs.
Mutation and effect tokens are rejected. Each export captures its registrations
and uses a private Torchax environment. Precision settings are scoped to export.
"""

import contextlib
from typing import Any, Callable, NamedTuple

import jax
import torch
import torchax
from torch.export.graph_signature import InputKind, OutputKind
from torchax import decompositions
from torchax import export as torchax_export

from iree_torchax import _torchax_registry

__all__ = [
    "JaxProgram",
    "exported_program_to_jax",
    "exported_program_to_stablehlo",
]


class JaxProgram(NamedTuple):
    constant_names: list[str]
    constants: list[Any]
    mutable_names: list[str]
    mutable: list[Any]
    func: Callable


def exported_program_to_jax(exported_program: torch.export.ExportedProgram):
    """Returns a JaxProgram for a stateless exported program.

    `func(constants, (), inputs)` returns `((), outputs)`. Constants follow the
    graph signature and tied tensors share one entry. The mutable fields are
    empty. Callers tracing this function choose their own JAX settings.
    """
    return _prepare(exported_program)[1]


def exported_program_to_stablehlo(exported_program: torch.export.ExportedProgram):
    """Returns `(program, artifact)` with symbolic input shapes preserved.

    `artifact` is a jax.export.Exported. Its entry takes the constants followed
    by user inputs, and returns the user outputs. Export temporarily enables
    64-bit types and full-precision f32 matmuls.
    """
    with _torch_semantics():
        prepared, program = _prepare(exported_program)
        # The avals come from the program the interpreter runs.
        inputs = torchax_export.extract_avals(prepared)
        # main keeps every argument, unused constants included, in program order.
        artifact = jax.export.export(jax.jit(program.func, keep_unused=True))(
            program.constants, program.mutable, tuple(inputs)
        )
    return program, artifact


@contextlib.contextmanager
def _torch_semantics():
    """JAX settings for tracing Torch programs: 64-bit types stay 64-bit, and
    f32 matmuls ask for full precision, torch's default matmul precision."""
    with jax.enable_x64(True), jax.default_matmul_precision("highest"):
        yield


def _by_overload(table):
    """`table` keyed by individual overloads; a packet's rule applies to each of
    its overloads unless the table also names the overload."""
    rules = {}
    for op, rule in table.items():
        if isinstance(op, torch._ops.OpOverloadPacket):
            for name in op.overloads():
                rules.setdefault(getattr(op, name), rule)
    rules.update(
        (op, rule)
        for op, rule in table.items()
        if not isinstance(op, torch._ops.OpOverloadPacket)
    )
    return rules


def _decompositions(entries):
    """Applies registered decompositions and keeps registered composites whole."""
    table = torch.export.default_decompositions()
    for op, rule in _by_overload(decompositions.DECOMPOSITIONS).items():
        table[op] = rule
    for entry in entries:
        if entry.kind == _torchax_registry.DECOMPOSITION:
            table[entry.key] = entry.handler
    for entry in entries:
        if entry.kind == _torchax_registry.COMPOSITE and isinstance(
            entry.key, torch._ops.OpOverload
        ):
            table.pop(entry.key, None)
    return table


def _to_jax(value):
    def leaf(item):
        if isinstance(item, torchax.tensor.Tensor):
            return item.jax()
        return item

    return jax.tree_util.tree_map(leaf, value)


class _Interpreter(torch.fx.Interpreter):
    """Runs ops through the overrides of `env`, the composites included."""

    def __init__(self, graph_module, env):
        super().__init__(graph_module)
        self._env = env

    def call_function(self, target, args, kwargs):
        # The environment's handlers, captured at preparation, resolve the
        # op; Torchax's global registry may have changed since.
        override = _torchax_registry.lookup(self._env._ops, target)
        if override is None:
            if isinstance(target, torch._ops.OpOverload):
                raise NotImplementedError(
                    f"Torchax has no lowering for {target}; "
                    "register_decomposition or register_composite can "
                    "provide one"
                )
            return super().call_function(target, args, kwargs)
        if not override.is_user_defined:
            # An exported program has no random state for Torchax's draws
            # to consume.
            if torch.Tag.nondeterministic_seeded in getattr(target, "tags", ()):
                raise NotImplementedError(
                    f"{target} draws random numbers, and an exported "
                    "program has no contract for the random state it "
                    "would consume"
                )
        if override.needs_env:
            kwargs = {**kwargs, "env": self._env}
        if override.is_jax_function:
            return _to_jax(override.func(*args, **kwargs))
        # Torch handlers run inside the environment their tensors belong to.
        with self._env:
            args, kwargs = self._env.j2t_iso((args, kwargs))
            return _to_jax(override.func(*args, **kwargs))


def _prepare(exported_program: torch.export.ExportedProgram):
    """Returns the decomposed program and its `JaxProgram`."""
    entries = _torchax_registry.snapshot()
    program = exported_program.run_decompositions(_decompositions(entries))
    if any(
        spec.kind == InputKind.TOKEN for spec in program.graph_signature.input_specs
    ):
        raise NotImplementedError("effect tokens are not supported")
    signature = program.graph_signature
    if signature.user_inputs_to_mutate:
        raise NotImplementedError(
            "mutated user inputs are not supported: "
            f"{sorted(signature.user_inputs_to_mutate.values())}"
        )

    # The placeholders take state and user inputs in the order of the input
    # specs, which decomposition need not keep grouped.
    values = dict(program.state_dict)
    values.update(program.constants)
    names = []
    # Per placeholder, the index of its state in `names`, or None for the next
    # user input.
    slots = []
    # Placeholders of one tensor, such as a tied weight, share its state.
    index_of_tensor = {}
    for spec in signature.input_specs:
        if spec.kind == InputKind.USER_INPUT:
            slots.append(None)
            continue
        if spec.kind not in (
            InputKind.PARAMETER,
            InputKind.BUFFER,
            InputKind.CONSTANT_TENSOR,
        ):
            raise NotImplementedError(
                f"program input {spec.target} of kind {spec.kind.name} is not "
                "supported"
            )
        index = index_of_tensor.setdefault(id(values[spec.target]), len(names))
        if index == len(names):
            names.append(spec.target)
        slots.append(index)
    user_input_count = slots.count(None)
    states = [values[name] for name in names]
    output_results = []
    for position, spec in enumerate(signature.output_specs):
        if spec.kind != OutputKind.USER_OUTPUT:
            raise NotImplementedError(
                f"program output {position} of kind {spec.kind.name} is not supported"
            )
        output_results.append(position)
    env = torchax.tensor.Environment()
    states = env.t2j_copy(states)
    _torchax_registry.install_snapshot(env, entries)
    constants = states

    def func(constants, mutable, inputs):
        if len(inputs) != user_input_count:
            raise TypeError(
                f"the program takes {user_input_count} inputs, not {len(inputs)}"
            )
        if len(constants) != len(states) or len(mutable):
            raise TypeError("expected the program constants and no mutable state")
        user_inputs = iter(inputs)
        placeholders = [
            next(user_inputs) if slot is None else constants[slot] for slot in slots
        ]
        results = _Interpreter(program.graph_module, env).run(
            *placeholders, enable_io_processing=False
        )
        return (), tuple(results[i] for i in output_results)

    return program, JaxProgram(names, constants, [], [], func)
