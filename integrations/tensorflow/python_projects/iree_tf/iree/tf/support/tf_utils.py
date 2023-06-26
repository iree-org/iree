# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utilities interop with TensorFlow."""

import os
import random
import re
from typing import Any, Callable, Mapping, Optional, Sequence, Set, Tuple, Union

import iree.runtime
import numpy as np
import tensorflow.compat.v2 as tf
from absl import logging

InputGeneratorType = Callable[[Sequence[int], Union[tf.DType, np.dtype]], np.ndarray]


def set_random_seed(seed: int = 0) -> None:
    """Set random seed for tf, np and random."""
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def uniform(
    shape: Sequence[int],
    dtype: Union[tf.DType, np.dtype] = np.float32,
    low: float = -1.0,
    high: float = 1.0,
) -> np.ndarray:
    """np.random.uniform with simplified API and dtype and bool support."""
    # pytype doesn't understand the ternary with tf as "Any"
    dtype = (
        dtype.as_numpy_dtype if isinstance(dtype, tf.DType) else dtype
    )  # pytype: disable=attribute-error
    if dtype == bool:
        return np.random.choice(2, shape).astype(bool)
    else:
        values = np.random.uniform(size=shape, low=low, high=high)
        if np.issubdtype(dtype, np.integer):
            values = np.round(values)
        return values.astype(dtype)


def ndarange(
    shape: Sequence[int], dtype: Union[tf.DType, np.dtype] = np.float32
) -> np.ndarray:
    """np.ndarange for arbitrary input shapes."""
    # pytype doesn't understand the ternary with tf as "Any"
    dtype = (
        dtype.as_numpy_dtype if isinstance(dtype, tf.DType) else dtype
    )  # pytype: disable=attribute-error
    return np.arange(np.prod(shape), dtype=dtype).reshape(shape)


def random_permutation(
    shape: Sequence[int], dtype: Union[tf.DType, np.dtype] = np.float32
) -> np.ndarray:
    """Returns a random permutation of [0, np.prod(shape))."""
    values = ndarange(shape, dtype)
    np.random.shuffle(values)
    return values


def apply_function(values, function):
    """Applies 'function' recursively to the inputted values."""
    if isinstance(values, list):
        return [apply_function(v, function) for v in values]
    elif isinstance(values, tuple):
        return tuple(apply_function(v, function) for v in values)
    elif isinstance(values, Mapping):
        return {k: apply_function(v, function) for k, v in values.items()}
    else:
        return function(values)


def generate_inputs(
    spec,  # Union[Sequence[tf.TensorSpec], tf.TensorSpec]
    input_generator: InputGeneratorType,
) -> Sequence[np.ndarray]:
    """Generates inputs for a given input signature using 'input_generator'."""
    make_static = lambda shape: [dim if dim is not None else 2 for dim in shape]
    generate = lambda spec: input_generator(make_static(spec.shape), spec.dtype)
    return apply_function(spec, generate)


def convert_to_numpy(values: Any) -> Any:
    """Converts any tf.Tensor, int, float, bool or list values to numpy."""
    return apply_function(values, iree.runtime.normalize_value)


def to_mlir_type(dtype: np.dtype) -> str:
    """Returns a string that denotes the type 'dtype' in MLIR style."""
    if not isinstance(dtype, np.dtype):
        # Handle np.int8 _not_ being a dtype.
        dtype = np.dtype(dtype)
    bits = dtype.itemsize * 8
    if np.issubdtype(dtype, np.integer):
        return f"i{bits}"
    elif np.issubdtype(dtype, np.floating):
        return f"f{bits}"
    else:
        raise TypeError(f"Expected integer or floating type, but got {dtype}")


def get_shape_and_dtype(array: np.ndarray, allow_non_mlir_dtype: bool = False) -> str:
    shape_dtype = [str(dim) for dim in list(array.shape)]
    if np.issubdtype(array.dtype, np.number):
        shape_dtype.append(to_mlir_type(array.dtype))
    elif allow_non_mlir_dtype:
        shape_dtype.append(f"<dtype '{array.dtype}'>")
    else:
        raise TypeError(f"Expected integer or floating type, but got {array.dtype}")
    return "x".join(shape_dtype)


def save_input_values(
    inputs: Sequence[np.ndarray], artifacts_dir: Optional[str] = None
) -> str:
    """Saves input values with IREE tools format if 'artifacts_dir' is set."""
    result = []
    for array in inputs:
        shape_dtype = get_shape_and_dtype(array)
        values = " ".join([str(x) for x in array.flatten()])
        result.append(f"{shape_dtype}={values}")
    result = "\n".join(result)
    if artifacts_dir is not None:
        inputs_path = os.path.join(artifacts_dir, "inputs.txt")
        logging.info("Saving IREE input values to: %s", inputs_path)
        with open(inputs_path, "w") as f:
            f.write(result)
            f.write("\n")
    return result


def remove_special_characters(value: str) -> str:
    """Replaces special characters with '_' while keeping instances of '__'."""
    normalized_parts = []
    for part in value.split("__"):
        part = re.sub(r"[^a-zA-Z0-9_]", "_", part)  # Remove special characters.
        part = re.sub(r"_+", "_", part)  # Remove duplicate "_".
        part = part.strip("_")  # Don't end or start in "_".
        normalized_parts.append(part)
    return "__".join(normalized_parts)


def is_complex(tensors: Union[Sequence[tf.TensorSpec], tf.TensorSpec]) -> bool:
    if isinstance(tensors, Sequence):
        for tensor in tensors:
            if is_complex(tensor):
                return True
        return False
    else:
        return tensors.dtype.is_complex  # pytype: disable=attribute-error


def _complex_wrapper(function):
    """Wraps a tf.function to allow compiling functions of complex numbers."""

    def decorator(*args, **kwargs):
        inputs = []
        for real, imag in zip(args[::2], args[1::2]):
            inputs.append(tf.complex(real, imag))
        result = function(*inputs, **kwargs)
        return tf.math.real(result), tf.math.imag(result)

    return decorator


def rewrite_complex_signature(function, signature: Sequence[tf.TensorSpec]):
    """Compatibility layer for testing complex numbers."""
    if not all([spec.dtype.is_complex for spec in signature]):
        raise NotImplementedError(
            "Signatures with mixed complex and non-complex "
            "tensor specs are not supported."
        )

    # Rewrite the signature, replacing all complex tensors with pairs of real
    # and imaginary tensors.
    real_imag_signature = []
    for spec in signature:
        new_dtype = tf.float32 if spec.dtype.size == 8 else tf.float64
        real_imag_signature.append(tf.TensorSpec(spec.shape, new_dtype))
        real_imag_signature.append(tf.TensorSpec(spec.shape, new_dtype))

    return _complex_wrapper(function), real_imag_signature


def make_dims_dynamic(spec: tf.TensorSpec) -> tf.TensorSpec:
    """Gives a tf.TensorSpec dynamic dims."""
    return tf.TensorSpec([None] * len(spec.shape), spec.dtype)


def check_same(
    ref: Any, tar: Any, rtol: float, atol: float
) -> Tuple[bool, Union[str, None]]:
    """Checks that ref and tar have identical datastructures and values."""
    # Check for matching types.
    if not isinstance(tar, type(ref)):
        error = (
            "Expected ref and tar to have the same type but got "
            f"'{type(ref)}' and '{type(tar)}'"
        )
        logging.error(error)
        return False, error

    if ref is None:
        # Nothing to compare (e.g. the called method had no outputs).
        return True, None

    # Recursive check for dicts.
    if isinstance(ref, dict):
        if ref.keys() != tar.keys():
            error = (
                "Expected ref and tar to have the same keys, but got "
                f"'{ref.keys()}' and '{tar.keys()}'"
            )
            logging.error(error)
            return False, error
        # Check that all of the dictionaries' values are the same.
        for key in ref:
            same, error = check_same(ref[key], tar[key], rtol, atol)
            if not same:
                return same, error

    # Recursive check for iterables.
    elif isinstance(ref, list) or isinstance(ref, tuple):
        if len(ref) != len(tar):
            error = (
                "Expected ref and tar to have the same length, but got "
                f"{len(ref)} and {len(tar)}"
            )
            logging.error(error)
            return False, error
        # Check that all of the iterables' values are the same.
        for i in range(len(ref)):
            same, error = check_same(ref[i], tar[i], rtol, atol)
            if not same:
                return same, error

    # Base check for numpy arrays.
    elif isinstance(ref, np.ndarray):
        # TODO(#5359): Simplify this and verify that the types are actually the same
        # Ignore np.bool != np.int8 because the IREE python runtime awkwardly
        # returns np.int8s instead of np.bools.
        if ref.dtype != tar.dtype and not (
            (ref.dtype == bool and tar.dtype == np.int8)
            or (ref.dtype == np.int8 and tar.dtype == bool)
        ):
            error = (
                "Expected ref and tar to have the same dtype, but got "
                f"'{ref.dtype}' and '{tar.dtype}'"
            )
            logging.error(error)
            return False, error

        if ref.size == tar.size == 0:
            return True, None

        if np.issubdtype(ref.dtype, np.floating):
            same = np.allclose(ref, tar, rtol=rtol, atol=atol, equal_nan=True)
            abs_diff = np.max(np.abs(ref - tar))
            rel_diff = np.max(np.abs(ref - tar) / np.max(np.abs(tar)))
            diff_string = (
                f"Max abs diff: {abs_diff:.2e}, atol: {atol:.2e}, "
                f"max relative diff: {rel_diff:.2e}, rtol: {rtol:.2e}"
            )
            if not same:
                error = (
                    "Floating point difference between ref and tar was too "
                    f"large. {diff_string}"
                )
                logging.error(error)
            else:
                error = None
                logging.info(
                    "Floating point difference between ref and tar was within "
                    "tolerance. %s",
                    diff_string,
                )
            return same, error
        elif np.issubdtype(ref.dtype, np.integer):
            same = np.array_equal(ref, tar)
            if not same:
                abs_diff = np.max(np.abs(ref - tar))
                error = (
                    "Expected array equality between ref and tar, but got "
                    f"a max elementwise difference of {abs_diff}"
                )
                logging.error(error)
            else:
                error = None
            return same, error
        else:
            return np.array_equal(ref, tar), None

    # Base check for native number types.
    elif isinstance(ref, (int, float)):
        return ref == tar, None

    # If outputs end up here then an extra branch for that type should be added.
    else:
        raise TypeError(f"Encountered results with unexpected type {type(ref)}")
    return True, None
