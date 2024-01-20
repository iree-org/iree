# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Dict, Optional

import json
import logging

import numpy as np

from ._binding import (
    _invoke_statics,
    ArgumentPacker,
    BufferUsage,
    HalBufferView,
    HalDevice,
    InvokeContext,
    MemoryType,
    VmContext,
    VmFunction,
    VmRef,
    VmVariantList,
)

from .array_interop import (
    map_dtype_to_element_type,
    DeviceArray,
)
from .flags import (
    FUNCTION_INPUT_VALIDATION,
)

__all__ = [
    "FunctionInvoker",
]


class Invocation:
    __slots__ = [
        "current_arg",
        "current_desc",
        "current_return_list",
        "current_return_index",
        "device",
    ]

    def __init__(self, device: HalDevice):
        self.device = device
        # Captured during arg/ret processing to emit better error messages.
        self.current_arg = None
        self.current_desc = None
        self.current_return_list = None
        self.current_return_index = 0

    def summarize_arg_error(self) -> str:
        if self.current_arg is None:
            return ""
        if isinstance(self.current_arg, np.ndarray):
            current_arg_repr = (
                f"ndarray({self.current_arg.shape}, {self.current_arg.dtype})"
            )
        else:
            current_arg_repr = repr(self.current_arg)
        return f"{repr(current_arg_repr)} with description {self.current_desc}"

    def summarize_return_error(self) -> str:
        if self.current_return_list is None:
            return ""
        try:
            vm_repr = f"{self.current_return_index}@{self.current_return_list}"
        except:
            vm_repr = "<error printing list item>"
        return f"{vm_repr} with description {self.current_desc}"


class FunctionInvoker:
    """Wraps a VmFunction, enabling invocations against it."""

    __slots__ = [
        "_vm_context",
        "_device",
        "_vm_function",
        "_abi_dict",
        "_arg_descs",
        "_arg_packer",
        "_ret_descs",
        "_has_inlined_results",
    ]

    def __init__(
        self,
        vm_context: VmContext,
        device: HalDevice,
        vm_function: VmFunction,
    ):
        self._vm_context = vm_context
        # TODO: Needing to know the precise device to allocate on here is bad
        # layering and will need to be fixed in some fashion if/when doing
        # heterogenous dispatch.
        self._device = device
        self._vm_function = vm_function
        self._abi_dict = None
        self._arg_descs = None
        self._ret_descs = None
        self._has_inlined_results = False
        self._parse_abi_dict(vm_function)
        self._arg_packer = ArgumentPacker(_invoke_statics, self._arg_descs)

    @property
    def vm_function(self) -> VmFunction:
        return self._vm_function

    def __call__(self, *args, **kwargs):
        invoke_context = InvokeContext(self._device)
        arg_list = self._arg_packer.pack(invoke_context, args, kwargs)

        # Initialize the capacity to our total number of args, since we should
        # be below that when doing a flat invocation. May want to be more
        # conservative here when considering nesting.
        inv = Invocation(self._device)
        ret_descs = self._ret_descs

        ret_list = VmVariantList(len(ret_descs) if ret_descs is not None else 1)
        self._invoke(arg_list, ret_list)

        # Un-inline the results to align with reflection, as needed.
        reflection_aligned_ret_list = ret_list
        if self._has_inlined_results:
            reflection_aligned_ret_list = VmVariantList(1)
            reflection_aligned_ret_list.push_list(ret_list)
        returns = _extract_vm_sequence_to_python(
            inv, reflection_aligned_ret_list, ret_descs
        )
        return_arity = len(returns)
        if return_arity == 1:
            return returns[0]
        elif return_arity == 0:
            return None
        else:
            return tuple(returns)

    # Break out invoke so it shows up in profiles.
    def _invoke(self, arg_list, ret_list):
        self._vm_context.invoke(self._vm_function, arg_list, ret_list)

    def _parse_abi_dict(self, vm_function: VmFunction):
        reflection = vm_function.reflection
        abi_json = reflection.get("iree.abi")
        if abi_json is None:
            # It is valid to have no reflection data, and rely on pure dynamic
            # dispatch.
            logging.debug(
                "Function lacks reflection data. Interop will be limited: %r",
                vm_function,
            )
            return
        try:
            self._abi_dict = json.loads(abi_json)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Reflection metadata is not valid JSON: {abi_json}"
            ) from e
        try:
            self._arg_descs = self._abi_dict["a"]
            self._ret_descs = self._abi_dict["r"]
        except KeyError as e:
            raise RuntimeError(
                f"Malformed function reflection metadata: {reflection}"
            ) from e
        if not isinstance(self._arg_descs, list) or not isinstance(
            self._ret_descs, list
        ):
            raise RuntimeError(
                f"Malformed function reflection metadata structure: {reflection}"
            )

        # Detect whether the results are a slist/stuple/sdict, which indicates
        # that they are inlined with the function's results.
        if len(self._ret_descs) == 1:
            maybe_inlined = self._ret_descs[0]
            if maybe_inlined and maybe_inlined[0] in ["slist", "stuple", "sdict"]:
                self._has_inlined_results = True

    def __repr__(self):
        return repr(self._vm_function)


# VM to Python converters. All take:
#   inv: Invocation
#   vm_list: VmVariantList to read from
#   vm_index: Index in the vm_list to extract
#   desc: The ABI descriptor list (or None if in dynamic mode)
# Return the corresponding Python object.


def _vm_to_ndarray(inv: Invocation, vm_list: VmVariantList, vm_index: int, desc):
    # The descriptor for an ndarray is like:
    #   ["ndarray", "<dtype>", <rank>, <dim>...]
    #   ex: ['ndarray', 'i32', 1, 25948]
    buffer_view = vm_list.get_as_object(vm_index, HalBufferView)
    dtype_str = desc[1]
    try:
        dtype = ABI_TYPE_TO_DTYPE[dtype_str]
    except KeyError:
        _raise_return_error(inv, f"unrecognized dtype '{dtype_str}'")
    x = DeviceArray(
        inv.device, buffer_view, implicit_host_transfer=True, override_dtype=dtype
    )
    return x


def _vm_to_sdict(inv: Invocation, vm_list: VmVariantList, vm_index: int, desc):
    # The descriptor for an sdict is like:
    #   ['sdict', ['key1', value1], ...]
    sub_vm_list = vm_list.get_as_list(vm_index)
    item_keys = []
    item_descs = []
    for k, d in desc[1:]:
        item_keys.append(k)
        item_descs.append(d)
    py_items = _extract_vm_sequence_to_python(inv, sub_vm_list, item_descs)
    return dict(zip(item_keys, py_items))


def _vm_to_slist(inv: Invocation, vm_list: VmVariantList, vm_index: int, desc):
    # The descriptor for an slist is like:
    #   ['slist, item1, ...]
    sub_vm_list = vm_list.get_as_list(vm_index)
    item_descs = desc[1:]
    py_items = _extract_vm_sequence_to_python(inv, sub_vm_list, item_descs)
    return py_items


def _vm_to_stuple(inv: Invocation, vm_list: VmVariantList, vm_index: int, desc):
    return tuple(_vm_to_slist(inv, vm_list, vm_index, desc))


def _vm_to_scalar(type_bound: type):
    def convert(inv: Invocation, vm_list: VmVariantList, vm_index: int, desc):
        value = vm_list.get_variant(vm_index)
        if not isinstance(value, type_bound):
            raise ReturnError(
                f"expected an {type_bound} value but got {value.__class__}"
            )
        return value

    return convert


def _vm_to_pylist(inv: Invocation, vm_list: VmVariantList, vm_index: int, desc):
    # The descriptor for a pylist is like:
    #   ['pylist', element_type]
    sub_vm_list = vm_list.get_as_list(vm_index)
    element_type_desc = desc[1:]
    py_items = _extract_vm_sequence_to_python(
        inv, sub_vm_list, element_type_desc * len(sub_vm_list)
    )
    return py_items


VM_TO_PYTHON_CONVERTERS = {
    "ndarray": _vm_to_ndarray,
    "sdict": _vm_to_sdict,
    "slist": _vm_to_slist,
    "stuple": _vm_to_stuple,
    "py_homogeneous_list": _vm_to_pylist,
    # Scalars.
    "i8": _vm_to_scalar(int),
    "i16": _vm_to_scalar(int),
    "i32": _vm_to_scalar(int),
    "i64": _vm_to_scalar(int),
    "f16": _vm_to_scalar(float),
    "f32": _vm_to_scalar(float),
    "f64": _vm_to_scalar(float),
    "bf16": _vm_to_scalar(float),
}

ABI_TYPE_TO_DTYPE = {
    # TODO: Others.
    "f32": np.float32,
    "i32": np.int32,
    "i64": np.int64,
    "f64": np.float64,
    "i16": np.int16,
    "i8": np.int8,
    "i1": np.bool_,
}

# When we get an ndarray as an argument and are implicitly mapping it to a
# buffer view, flags for doing so.
IMPLICIT_BUFFER_ARG_MEMORY_TYPE = MemoryType.DEVICE_LOCAL
IMPLICIT_BUFFER_ARG_USAGE = BufferUsage.DEFAULT | BufferUsage.MAPPING


def _is_ndarray_descriptor(desc):
    return desc and desc[0] == "ndarray"


def _is_0d_ndarray_descriptor(desc):
    # Example: ["ndarray", "f32", 0]
    return desc and desc[0] == "ndarray" and desc[2] == 0


def _cast_scalar_to_ndarray(inv: Invocation, x, desc):
    # Example descriptor: ["ndarray", "f32", 0]
    dtype_str = desc[1]
    try:
        dtype = ABI_TYPE_TO_DTYPE[dtype_str]
    except KeyError:
        _raise_argument_error(inv, f"unrecognized dtype '{dtype_str}'")
    return dtype(x)


class ArgumentError(ValueError):
    pass


class ReturnError(ValueError):
    pass


def _raise_argument_error(inv: Invocation, summary: str, e: Optional[Exception] = None):
    new_e = ArgumentError(
        f"Error passing argument: {summary} "
        f"(while encoding argument {inv.summarize_arg_error()})"
    )
    if e:
        raise new_e from e
    else:
        raise new_e


def _raise_return_error(inv: Invocation, summary: str, e: Optional[Exception] = None):
    new_e = ReturnError(
        f"Error processing function return: {summary} "
        f"(while decoding return {inv.summarize_return_error()})"
    )
    if e:
        raise new_e from e
    else:
        raise new_e


def _extract_vm_sequence_to_python(inv: Invocation, vm_list, descs):
    vm_list_arity = len(vm_list)
    if descs is None:
        descs = [None] * vm_list_arity
    elif vm_list_arity != len(descs):
        _raise_return_error(
            inv, f"mismatched return arity: {vm_list_arity} vs {len(descs)}"
        )
    results = []
    for vm_index, desc in zip(range(vm_list_arity), descs):
        inv.current_return_list = vm_list
        inv.current_return_index = vm_index
        inv.current_desc = desc
        if desc is None:
            # Dynamic (non reflection mode).
            converted = vm_list.get_variant(vm_index)
            # Special case: Upgrade HalBufferView to a DeviceArray. We do that here
            # since this is higher level and it preserves layering. Note that
            # the reflection case also does this conversion.
            if isinstance(converted, VmRef):
                converted_buffer_view = converted.deref(HalBufferView, True)
                if converted_buffer_view:
                    converted = DeviceArray(
                        inv.device, converted_buffer_view, implicit_host_transfer=True
                    )
        else:
            # Known type descriptor.
            vm_type = desc if isinstance(desc, str) else desc[0]
            try:
                converter = VM_TO_PYTHON_CONVERTERS[vm_type]
            except KeyError:
                _raise_return_error(inv, f"cannot map VM type to Python: {vm_type}")
            try:
                converted = converter(inv, vm_list, vm_index, desc)
            except ReturnError:
                raise
            except Exception as e:
                _raise_return_error(
                    inv, f"exception converting from VM type to Python", e
                )
        results.append(converted)
    return results
