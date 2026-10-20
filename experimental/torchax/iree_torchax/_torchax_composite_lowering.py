# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Operand-signature composite lowering for the pinned JAX version.

JAX prepends module dimension arguments to decomposition functions. IREE
handlers use the declared tensor operands, so a wrapper reconstructs those
dimensions from operand shapes without changing the composite signature.
This module is loaded only when a composite is first called.
"""

import functools
import itertools
import numpy as np
from jax._src.hashable_array import HashableArray
from jax._src import api_util as _api_util
from jax._src import config as _jax_config
from jax._src import core as _jax_core
from jax._src import sharding_impls as _sharding_impls
from jax._src import source_info_util as _source_info_util
from jax._src import tree_util as _tree_util
from jax._src import util as _jax_util
from jax._src.export import shape_poly as _shape_poly
from jax._src.interpreters import mlir as _mlir
from jax._src.interpreters import partial_eval as _pe
from jax._src.lax import lax as _lax
from jax._src.lib.mlir import ir as _ir
from jax._src.lib.mlir.dialects import func as _func
from jax._src.lib.mlir.dialects import hlo as _hlo


@functools.cache
def _composite_primitive():
    """Shares one adapter primitive across composite names."""
    primitive = _jax_core.Primitive("iree_torch_composite")
    primitive.multiple_results = True
    primitive.def_impl(_lax._composite_impl)
    primitive.def_abstract_eval(_lax._composite_abstract_eval)
    primitive.to_lojax = functools.partial(_pe._eval_jaxpr_to_lojax, primitive)
    _mlir.register_lowering(primitive, _composite_lowering)
    return primitive


def composite(decomposition, name, version):
    """As jax.lax.composite, binding the adapter primitive."""

    @functools.wraps(decomposition)
    def bind(*args, **kwargs):
        debug_info = _api_util.debug_info("composite", decomposition, args, {})
        flat_args, in_tree = _tree_util.tree_flatten(args)
        in_avals = tuple(_jax_core.typeof(x) for x in flat_args)
        closed_jaxpr, consts, out_tree = _lax._trace_composite_to_jaxpr(
            functools.partial(decomposition, **kwargs),
            in_tree,
            in_avals,
            name,
            debug_info,
        )
        attributes = []
        for key, value in kwargs.items():
            leaves, treedef = _tree_util.tree_flatten(value)
            attributes.append(
                (
                    key,
                    tuple(
                        HashableArray(value) if isinstance(value, np.ndarray) else value
                        for value in leaves
                    ),
                    treedef,
                )
            )
        out_flat = _composite_primitive().bind(
            *_jax_core.auto_insert_reshard(*consts, *flat_args),
            name=name,
            attributes=tuple(attributes),
            version=version,
            call_jaxpr=closed_jaxpr,
        )
        return _tree_util.tree_unflatten(out_tree, out_flat)

    return bind


def _operand_decomposition(module_context, body, in_avals, operands):
    """Returns a function over exactly |operands| that calls |body| with the
    dimension values recovered from the operand shapes."""
    dim_vars = module_context.shape_poly_state.dim_vars
    signature = _ir.FunctionType.get([v.type for v in operands], body.type.results)
    wrapper = _func.FuncOp(
        f"{body.name.value}_operands",
        signature,
        ip=_ir.InsertionPoint.at_block_begin(module_context.module.body),
    )
    wrapper.attributes["sym_visibility"] = _ir.StringAttr.get("private")
    module_context.symbol_table.insert(wrapper)
    entry = wrapper.add_entry_block()
    with _ir.InsertionPoint(entry):
        # As jax.export's wrapper of main: a context without dimension
        # variables lowers the dimension computation.
        dim_context = _mlir.ModuleContext(
            backend=None,
            platforms=["cpu"],
            axis_context=_sharding_impls.ShardingContext(0),
            keepalives=[],
            channel_iterator=itertools.count(1),
            host_callbacks=[],
            module=module_context.module,
            context=module_context.context,
            # Helpers it lowers take names unique in the module.
            symbol_table=module_context.symbol_table,
            lowering_parameters=_mlir.LoweringParameters(
                global_constant_computation=True,
                for_export=True,
                hoist_constants_as_args=False,
                export_ignore_forward_compatibility=_jax_config.export_ignore_forward_compatibility.value,
            ),
        )
        ctx = _mlir.LoweringRuleContext(
            module_context=dim_context,
            name_stack=_source_info_util.new_name_stack(),
            traceback=None,
            primitive=None,
            avals_in=in_avals,
            avals_out=None,
            tokens_in=_mlir.TokenSet(),
            tokens_out=None,
            const_lowering={},
        )
        values = _mlir.lower_fun(
            functools.partial(
                _shape_poly.compute_dim_vars_from_arg_shapes,
                in_avals,
                args_kwargs_tree=_tree_util.tree_structure((tuple(in_avals), {})),
            ),
            multiple_results=True,
        )(ctx, *entry.arguments)
        flat_values, _ = _mlir.ir_tree_registry.flatten(values)
        known = dict(zip(_shape_poly.all_dim_vars(in_avals), flat_values))
        dim_args = []
        if module_context.shape_poly_state.has_platform_index_argument:
            # The caller checks that this selector is unused by the body.
            dim_args.append(_mlir.ir_constant(np.zeros((), np.int32)))
        for var, arg_type in zip(dim_vars, body.type.inputs[len(dim_args) :]):
            value = known.get(var)
            if value is None:
                index = len(dim_args)
                if list(body.body.blocks[0].arguments[index].uses):
                    raise NotImplementedError(
                        f"composite dimension {var} cannot be recovered from its operands"
                    )
                # Unused module dimensions need an argument value but do not
                # affect this decomposition's computation.
                value = _mlir.ir_constant(np.zeros((), np.int64))
            if value.type != arg_type:
                value = _hlo.convert(arg_type, value)
            dim_args.append(value)
        call = _func.CallOp(
            body.type.results,
            _ir.FlatSymbolRefAttr.get(body.name.value),
            [*dim_args, *entry.arguments],
        )
        _func.ReturnOp(call.results)
    return wrapper


def _can_recover_dimensions(module_context, body, in_avals):
    """Checks recoverability before creating any wrapper IR."""
    state = module_context.shape_poly_state
    prefix = int(state.has_platform_index_argument)
    arguments = body.body.blocks[0].arguments
    if prefix and list(arguments[0].uses):
        return False
    try:
        solution, _, _ = _shape_poly.solve_dim_vars(
            tuple(in_avals),
            args_kwargs_tree=_tree_util.tree_structure((tuple(in_avals), {})),
        )
    except ValueError:
        return False
    return not any(
        var not in solution and list(arguments[prefix + i].uses)
        for i, var in enumerate(state.dim_vars)
    )


def _composite_lowering(ctx, *args, name, attributes, version, call_jaxpr):
    shape_poly_state = ctx.module_context.shape_poly_state
    if not (shape_poly_state.dim_vars or shape_poly_state.has_platform_index_argument):
        return _lax._composite_lowering(
            ctx,
            *args,
            name=name,
            attributes=attributes,
            version=version,
            call_jaxpr=call_jaxpr,
        )
    const_args_and_avals = _jax_core.jaxpr_const_args(call_jaxpr)
    const_args, const_avals = _jax_util.unzip2(const_args_and_avals)
    const_values = tuple(
        _mlir.ir_constants(c, const_lowering=ctx.const_lowering, aval=aval)
        for c, aval in const_args_and_avals
    )
    in_avals = (*const_avals, *ctx.avals_in)
    body, _, _ = _mlir.lower_called_computation(
        name,
        call_jaxpr,
        ctx.module_context,
        len(const_args),
        in_avals,
        ctx.avals_out,
        ctx.tokens_in,
    )
    if not _can_recover_dimensions(ctx.module_context, body, in_avals):
        return _lax._composite_lowering(
            ctx,
            *args,
            name=name,
            attributes=attributes,
            version=version,
            call_jaxpr=call_jaxpr,
        )
    operands, _ = _mlir.ir_tree_registry.flatten(const_values + args)
    decomposition = _operand_decomposition(ctx.module_context, body, in_avals, operands)
    composite_attrs = {}
    for key, leaves, treedef in attributes:
        value = treedef.unflatten(leaves)
        if value is not None:
            composite_attrs[key] = _mlir.ir_attribute(value)
    return _hlo.CompositeOp(
        body.type.results,
        operands,
        name=_ir.StringAttr.get(name),
        decomposition=_ir.FlatSymbolRefAttr.get(decomposition.name.value),
        composite_attributes=_ir.DictAttr.get(composite_attrs),
        version=_mlir.i32_attr(version),
    ).results
