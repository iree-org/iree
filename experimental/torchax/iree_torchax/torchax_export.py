# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Exports torch programs through torchax with their buffer mutations.

`torchax.export.exported_program_to_jax` drops the new values of mutated
buffers, so a KV-cache update is lost. `exported_program_to_jax` here returns
them instead. Jit the function with `donate_argnums=(1,)`: JAX marks each
donated buffer with `tf.aliasing_output`, which IREE turns into
`iree.abi.output`. Storage reuse remains subject to IREE's ownership checks.

`exported_program_to_stablehlo` does the same for programs exported with
dynamic shapes: it exports with the symbolic input shapes and their
constraints, so one artifact serves every length the constraints allow. State
goes in and out explicitly: `main(*constants, *state, *inputs)` returns
`(*new_state, *outputs)`, and the caller passes the returned state to the next
call.

`externalize_parameters` then moves the immutable weights out of the entry
signature into parameter globals, loaded from an IRPA archive at run time.

Compiled with `--iree-execution-model=async-external`, the module exports
only `main`, which takes a wait fence and a signal fence after its inputs:
it starts once the wait fence is reached, and its outputs, the donated state
included, are ready once the signal fence is. A failed wait fence fails the
signal fence.

Per-export composite and decomposition registrations apply without changing
Torchax's global registry. Import the functions through `iree_torchax`.

Tested with torchax 0.0.14.dev20261007 and jax 0.12.0.dev20261016, the first
JAX whose `jax.lax.composite` is valid under symbolic shapes.
"""

import contextlib
from typing import Any, Callable, NamedTuple, Sequence

import jax
import numpy as np
import torch
import torchax
from torch.export._remove_effect_tokens_pass import _remove_effect_tokens
from torch.export.graph_signature import InputKind, OutputKind
from torchax import decompositions
from torchax import export as torchax_export
from torchax.ops import mappings

from iree.compiler import ir
from iree.compiler.dialects import func as func_d
from iree.compiler.dialects import util
from iree_torchax import _torchax_registry

__all__ = [
    "JaxProgram",
    "exported_program_to_jax",
    "exported_program_to_stablehlo",
    "externalize_parameters",
    "save_parameters",
]


class JaxProgram(NamedTuple):
    constant_names: list[str]
    constants: list[Any]
    mutable_names: list[str]
    mutable: list[Any]
    func: Callable


def exported_program_to_jax(
    exported_program: torch.export.ExportedProgram, *, copy_constants: bool = False
):
    """Converts `exported_program` to JAX, keeping mutated buffers and parameters.

    The constants are JAX arrays on the storage of the program's tensors, so
    the weights are never copied. A constant whose storage JAX cannot alias,
    one not contiguous and 64-byte aligned, raises `ValueError` unless
    `copy_constants` is set. The mutable state is copied, since running `func`
    donates it.

    `func(constants, mutable, inputs)` returns `(new_mutable, outputs)`, where
    `new_mutable` holds the updated buffers and parameters in the order of
    `mutable`.
    Operations with a registered composite stay whole in the exported program.

    The caller traces `func` under its own JAX configuration. Torch's int64
    and f64 types and its f32 matmul precision need
    `jax.enable_x64(True)` and `jax.default_matmul_precision("highest")`,
    which `exported_program_to_stablehlo` sets itself.
    """
    program = _prepare(exported_program)[1]
    constants = []
    with _torch_semantics():
        for name, tensor in zip(program.constant_names, program.constants):
            array, reason = _share(tensor)
            if array is None:
                if not copy_constants:
                    raise ValueError(
                        f"constant {name} {tensor.dtype}{list(tensor.shape)} cannot "
                        f"share its storage with JAX: {reason}; pass "
                        "copy_constants=True to copy it"
                    )
                array = mappings.t2j(tensor, False)
            constants.append(array)
        mutable = [mappings.t2j(tensor, False) for tensor in program.mutable]
    return program._replace(constants=constants, mutable=mutable)


def exported_program_to_stablehlo(
    exported_program: torch.export.ExportedProgram, *, donate_state: bool = True
):
    """Exports `exported_program` with symbolic input shapes, keeping mutations.

    Returns `(program, artifact)`: the `jax.export.Exported` of `program.func`
    with mutable buffers donated by default, and `program` as
    `exported_program_to_jax` returns it except that its constants and initial
    state are the exported program's own tensors. The export reads no weight
    data, so weights on the meta device export too; `save_parameters` writes
    the constants. The user inputs keep the dynamic
    dimensions and bounds of `exported_program`; constants and state keep their
    shapes.

    Set `donate_state=False` when the caller retains old state values. With
    donation enabled, the caller gives up the input state and must use the
    returned state for subsequent calls.

    The caller threads the state through calls:

        import iree_torchax

        program, artifact = iree_torchax.exported_program_to_stablehlo(exported)
        vmfb = compile_str(
            str(artifact.mlir_module()), target_backends=["llvm-cpu"])
        main = ...  # `main` of the loaded vmfb
        state = list(program.mutable)
        for inputs in calls:
            *state, output = main(*program.constants, *state, *inputs)
    """
    with _torch_semantics():
        prepared, program = _prepare(exported_program)
        # The avals come from the program the interpreter runs.
        inputs = torchax_export.extract_avals(prepared)

        def avals(tensors):
            return [
                jax.ShapeDtypeStruct(tuple(t.shape), mappings.t2j_dtype(t.dtype))
                for t in tensors
            ]

        # main keeps every argument, unused constants included, in program order.
        artifact = jax.export.export(
            jax.jit(
                program.func,
                donate_argnums=(1,) if donate_state else (),
                keep_unused=True,
            )
        )(avals(program.constants), avals(program.mutable), tuple(inputs))
    return program, artifact


def save_parameters(program: JaxProgram, path) -> None:
    """Writes `program`'s constants to the IRPA archive `path` under the names
    `externalize_parameters` binds, straight from their storage."""
    from iree.runtime import save_archive_file

    entries = {}
    for name, value in zip(program.constant_names, program.constants):
        if not isinstance(value, torch.Tensor):
            entries[name] = np.asarray(value).reshape(-1).view(np.uint8)
            continue
        if value.device.type != "cpu":
            raise ValueError(f"constant {name} is on {value.device} and has no data")
        # A byte view writes bfloat16 and the float8 types, which NumPy lacks.
        entries[name] = (
            value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
        )
    save_archive_file(entries, path)


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
    """torch.export's, then torchax's, then the
    registered decompositions, keeping every registered composite whole."""
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
        # A compiled program cannot raise; a failed factorisation shows as
        # NaN or Inf, as it does in JAX.
        if (
            target is torch.ops.aten._linalg_check_errors.default
            and _torchax_registry.lookup(self._env._ops, target) is None
        ):
            return None
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


def _share(tensor: torch.Tensor):
    """Returns a JAX array on `tensor`'s storage, or why there can be none."""
    if tensor.device.type != "cpu" or tensor.layout != torch.strided:
        return None, f"it is a {tensor.layout} tensor on {tensor.device}"
    tensor = tensor.detach()
    dtype = np.dtype(mappings.t2j_dtype(tensor.dtype))
    try:
        view = tensor.numpy()
    except TypeError:
        # NumPy lacks bfloat16 and the float8 types; reinterpret their bits.
        bits = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
        view = tensor.view(bits[tensor.element_size()]).numpy().view(dtype)
    # An uncommitted array, unlike a DLPack one, adds no sharding to lowerings.
    array = jax.device_put(view, may_alias=True)
    # JAX copies rather than fail for storage it cannot alias, such as one not
    # contiguous or not 64-byte aligned.
    if tensor.numel() and array.unsafe_buffer_pointer() != tensor.data_ptr():
        return None, "JAX cannot alias it, as it is not contiguous and 64-byte aligned"
    return array, None


def _prepare(exported_program: torch.export.ExportedProgram):
    """Returns the decomposed program and its `JaxProgram`, whose constants and
    mutable state are the program's tensors."""
    entries = _torchax_registry.snapshot()
    program = exported_program.run_decompositions(_decompositions(entries))
    # Side-effecting operations, such as linear algebra error checks, thread
    # a token the interpreter does not pass.
    if any(
        spec.kind == InputKind.TOKEN for spec in program.graph_signature.input_specs
    ):
        program = _remove_effect_tokens(program)
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
    mutated = []
    mutation_results = []
    mutable_positions = []
    output_results = []
    for position, spec in enumerate(signature.output_specs):
        if spec.kind in (OutputKind.BUFFER_MUTATION, OutputKind.PARAMETER_MUTATION):
            state_position = index_of_tensor[id(values[spec.target])]
            if state_position in mutable_positions:
                raise NotImplementedError(
                    "multiple updates of aliased buffers are unsupported"
                )
            mutable_positions.append(state_position)
            mutated.append(spec.target)
            mutation_results.append(position)
        elif spec.kind == OutputKind.USER_OUTPUT:
            output_results.append(position)
        else:
            raise NotImplementedError(
                f"program output {position} of kind {spec.kind.name} is not "
                "supported"
            )
    constant_positions = [i for i in range(len(states)) if i not in mutable_positions]
    env = torchax.tensor.Environment()
    _torchax_registry.install_snapshot(env, entries)

    def func(constants, mutable, inputs):
        if len(inputs) != user_input_count:
            raise TypeError(
                f"the program takes {user_input_count} inputs, not {len(inputs)}"
            )
        if len(constants) != len(constant_positions) or len(mutable) != len(
            mutable_positions
        ):
            raise TypeError("expected the program constants and mutable state")
        ordered = [None] * len(states)
        for position, value in zip(constant_positions, constants):
            ordered[position] = value
        for position, value in zip(mutable_positions, mutable):
            ordered[position] = value
        user_inputs = iter(inputs)
        placeholders = [
            next(user_inputs) if slot is None else ordered[slot] for slot in slots
        ]
        results = _Interpreter(program.graph_module, env).run(
            *placeholders, enable_io_processing=False
        )
        new_mutable = tuple(results[i] for i in mutation_results)
        for name, old, new in zip(mutated, mutable, new_mutable):
            if old.shape != new.shape or old.dtype != new.dtype:
                raise ValueError(
                    f"state {name} changes from {old.dtype}{list(old.shape)} "
                    f"to {new.dtype}{list(new.shape)}; it cannot alias its update"
                )
        return new_mutable, tuple(results[i] for i in output_results)

    return program, JaxProgram(
        [names[i] for i in constant_positions],
        # Detached views share storage and read without autograd.
        [states[i].detach() for i in constant_positions],
        mutated,
        [states[i].detach() for i in mutable_positions],
        func,
    )


def externalize_parameters(
    module_text: str,
    parameter_names: Sequence[str],
    scope: str = "model",
    entry: str = "main",
) -> str:
    """Loads the leading arguments of `entry` from named parameters.

    Argument `i < len(parameter_names)` becomes a `util.global` bound to
    `#flow.parameter.named<scope::parameter_names[i]>`. The original function
    becomes private, and a new public `entry` with the remaining arguments
    loads the globals and calls it.
    """
    with ir.Context() as context, ir.Location.unknown():
        context.allow_unregistered_dialects = True
        module = ir.Module.parse(module_text)
        symbols = ir.SymbolTable(module.operation)
        callee = symbols[entry]
        callee_type = ir.FunctionType(
            ir.TypeAttr(callee.attributes["function_type"]).value
        )
        count = len(parameter_names)
        if count > len(callee_type.inputs):
            raise ValueError(
                f"@{entry} has {len(callee_type.inputs)} arguments, fewer than "
                f"the {count} parameters"
            )
        taken = set()

        def unused(base):
            # The table caches the module's symbols; `taken` adds those made since.
            name, suffix = base, 0
            while name in symbols or name in taken:
                name, suffix = f"{base}_{suffix}", suffix + 1
            taken.add(name)
            return name

        callee_name = unused(f"{entry}$with_parameters")
        # Callers within the module keep the full signature.
        ir.SymbolTable.replace_all_symbol_uses(entry, callee_name, module.operation)
        ir.SymbolTable.set_symbol_name(callee, callee_name)
        callee.attributes["sym_visibility"] = ir.StringAttr.get("private")
        arg_attrs = (
            list(callee.attributes["arg_attrs"])
            if "arg_attrs" in callee.attributes
            else None
        )
        res_attrs = (
            callee.attributes["res_attrs"] if "res_attrs" in callee.attributes else None
        )
        empty = ir.DictAttr.get({})
        if arg_attrs is not None:
            callee.attributes["arg_attrs"] = ir.ArrayAttr.get([empty] * len(arg_attrs))
        if res_attrs is not None:
            del callee.attributes["res_attrs"]

        with ir.InsertionPoint.at_block_begin(module.body):
            globals_ = []
            for name, type_ in zip(parameter_names, callee_type.inputs[:count]):
                global_ = util.GlobalOp(
                    unused(f"__parameter.{name}"),
                    type_,
                    sym_visibility="private",
                    initial_value=ir.Attribute.parse(
                        f'#flow.parameter.named<"{scope}"::"{name}"> : {type_}'
                    ),
                )
                globals_.append((global_.sym_name.value, type_))

        with ir.InsertionPoint(module.body):
            wrapper = func_d.FuncOp(
                entry,
                (callee_type.inputs[count:], callee_type.results),
            )
            if arg_attrs is not None:
                wrapper.attributes["arg_attrs"] = ir.ArrayAttr.get(arg_attrs[count:])
            if res_attrs is not None:
                wrapper.attributes["res_attrs"] = res_attrs
            block = wrapper.add_entry_block()
            with ir.InsertionPoint(block):
                loads = [
                    util.GlobalLoadOp(type_, symbol).result
                    for symbol, type_ in globals_
                ]
                call = func_d.CallOp(
                    callee_type.results,
                    ir.FlatSymbolRefAttr.get(callee_name),
                    loads + list(block.arguments),
                )
                func_d.ReturnOp(call.results)
        return str(module)
