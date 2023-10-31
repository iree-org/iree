# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Util functions/classes for jax primitive test harnesses."""

import contextlib
import functools
from typing import Optional
import warnings
import zlib
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import jax2tf
from jax.experimental.jax2tf.tests import jax2tf_limitations
from jax.experimental.jax2tf.tests import primitive_harness
import ml_dtypes
import numpy as np
import numpy.random as npr
import tensorflow as tf

_SUPPORTED_DTYPES = [np.float32]


def _harness_matches(harness, group_name, dtype, params):
    if harness.group_name != group_name:
        return False
    if dtype is not None and harness.dtype != dtype:
        return False
    for key, value in params.items():
        if harness.params.get(key, None) != value:
            return False
    return True


_CRASH_LIST_PARMS = []


_DEFAULT_TOLERANCE = {
    jax.dtypes.float0: 0,
    np.dtype(np.bool_): 0,
    np.dtype(ml_dtypes.int4): 0,
    np.dtype(np.int8): 0,
    np.dtype(np.int16): 0,
    np.dtype(np.int32): 0,
    np.dtype(np.int64): 0,
    np.dtype(ml_dtypes.uint4): 0,
    np.dtype(np.uint8): 0,
    np.dtype(np.uint16): 0,
    np.dtype(np.uint32): 0,
    np.dtype(np.uint64): 0,
    np.dtype(ml_dtypes.float8_e4m3b11fnuz): 1e-1,
    np.dtype(ml_dtypes.float8_e4m3fn): 1e-1,
    np.dtype(ml_dtypes.float8_e5m2): 1e-1,
    np.dtype(ml_dtypes.bfloat16): 1e-2,
    np.dtype(np.float16): 1e-3,
    np.dtype(np.float32): 1e-6,
    np.dtype(np.float64): 1e-15,
    np.dtype(np.complex64): 1e-6,
    np.dtype(np.complex128): 1e-15,
}


def _dtype(x):
    if hasattr(x, "dtype"):
        return x.dtype
    elif type(x) in jax.dtypes.python_scalar_dtypes:
        return np.dtype(jax.dtypes.python_scalar_dtypes[type(x)])
    else:
        return np.asarray(x).dtype


def tolerance(dtype, tol=None):
    tol = {} if tol is None else tol
    if not isinstance(tol, dict):
        return tol
    tol = {np.dtype(key): value for key, value in tol.items()}
    dtype = jax.dtypes.canonicalize_dtype(np.dtype(dtype))
    return tol.get(dtype, _DEFAULT_TOLERANCE[dtype])


def _assert_numpy_allclose(a, b, atol=None, rtol=None, err_msg=""):
    """Checks if two numpy arrays are all close given tolerances.

    Args:
      a: The array to check.
      b: The expected array.
      atol: Absolute tolerance.
      rtol: Relative tolerance.
      err_msg: The error message to print in case of failure.
    """
    if a.dtype == b.dtype == jax.dtypes.float0:
        np.testing.assert_array_equal(a, b, err_msg=err_msg)
        return
    custom_dtypes = [
        ml_dtypes.float8_e4m3b11fnuz,
        ml_dtypes.float8_e4m3fn,
        ml_dtypes.float8_e5m2,
        ml_dtypes.bfloat16,
    ]
    a = a.astype(np.float32) if a.dtype in custom_dtypes else a
    b = b.astype(np.float32) if b.dtype in custom_dtypes else b
    kw = {}
    if atol:
        kw["atol"] = atol
    if rtol:
        kw["rtol"] = rtol
    with np.errstate(invalid="ignore"):
        # TODO(phawkins): surprisingly, assert_allclose sometimes reports invalid
        # value errors. It should not do that.
        np.testing.assert_allclose(a, b, **kw, err_msg=err_msg)


@contextlib.contextmanager
def ignore_warning(**kw):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", **kw)
        yield


def _has_only_supported_dtypes(harness):
    return True
    if harness.dtype not in _SUPPORTED_DTYPES:
        return False

    for key, value in harness.params.items():
        if "dtype" in key and value not in _SUPPORTED_DTYPES:
            return False

    return True


def is_sequence(x):
    try:
        iter(x)
    except TypeError:
        return False
    else:
        return True


def primitives_parameterized(harnesses, *, one_containing: Optional[str] = None):
    """Decorator for tests. This is used to filter the tests.

    Args:
      harnesses: List of Harness objects to be filtered.
      one_containing: If set, only creates one test case for the provided name.

    Returns:
      A parameterized version of the test function with filtered set of harnesses.
    """

    def _filter_harness(harness):
        # TODO(b/295369536) Put a limitations system in place so what's not covered
        # is explicit.
        if not harness.params.get("enable_xla", True):
            return False

        if one_containing is not None and one_containing not in harness.fullname:
            return False

        if not _has_only_supported_dtypes(harness):
            return False

        for crash_item in _CRASH_LIST_PARMS:
            if _harness_matches(
                harness,
                crash_item["group_name"],
                crash_item["dtype"],
                crash_item["params"],
            ):
                return False

        return True

    harnesses = filter(_filter_harness, harnesses)

    return primitive_harness.parameterized(harnesses, include_jax_unimpl=False)


class JaxRunningTestCase(parameterized.TestCase):
    """A test case for JAX conversions."""

    def setUp(self):
        super().setUp()

        # We use the adler32 hash for two reasons.
        # a) it is deterministic run to run, unlike hash() which is randomized.
        # b) it returns values in int32 range, which RandomState requires.
        self._rng = npr.RandomState(zlib.adler32(self._testMethodName.encode()))

    def rng(self):
        return self._rng

    def assert_all_close(
        self,
        x,
        y,
        *,
        check_dtypes=True,
        atol=None,
        rtol=None,
        canonicalize_dtypes=True,
        err_msg="",
    ):
        """Assert that x and y, either arrays or nested tuples/lists, are close."""
        if isinstance(x, dict):
            self.assertIsInstance(y, dict)
            self.assertEqual(set(x.keys()), set(y.keys()))
            for k in x.keys():
                self.assert_all_close(
                    x[k],
                    y[k],
                    check_dtypes=check_dtypes,
                    atol=atol,
                    rtol=rtol,
                    canonicalize_dtypes=canonicalize_dtypes,
                    err_msg=err_msg,
                )
        elif is_sequence(x) and not hasattr(x, "__array__"):
            self.assertTrue(is_sequence(y) and not hasattr(y, "__array__"))
            self.assertEqual(len(x), len(y))
            for x_elt, y_elt in zip(x, y):
                self.assert_all_close(
                    x_elt,
                    y_elt,
                    check_dtypes=check_dtypes,
                    atol=atol,
                    rtol=rtol,
                    canonicalize_dtypes=canonicalize_dtypes,
                    err_msg=err_msg,
                )
        elif hasattr(x, "__array__") or np.isscalar(x):
            self.assertTrue(hasattr(y, "__array__") or np.isscalar(y))
            if check_dtypes:
                self.assert_dtypes_match(x, y, canonicalize_dtypes=canonicalize_dtypes)
            x = np.asarray(x)
            y = np.asarray(y)
            self.assert_arrays_all_close(
                x, y, check_dtypes=False, atol=atol, rtol=rtol, err_msg=err_msg
            )
        elif x == y:
            return
        else:
            raise TypeError((type(x), type(y)))

    def assert_arrays_all_close(
        self, x, y, *, check_dtypes=True, atol=None, rtol=None, err_msg=""
    ):
        """Assert that x and y are close (up to numerical tolerances)."""
        self.assertEqual(x.shape, y.shape)
        atol = max(tolerance(_dtype(x), atol), tolerance(_dtype(y), atol))
        rtol = max(tolerance(_dtype(x), rtol), tolerance(_dtype(y), rtol))

        _assert_numpy_allclose(x, y, atol=atol, rtol=rtol, err_msg=err_msg)

        if check_dtypes:
            self.assert_dtypes_match(x, y)

    def assert_dtypes_match(self, x, y, *, canonicalize_dtypes=True):
        if not jax.config.x64_enabled and canonicalize_dtypes:
            self.assertEqual(
                jax.dtypes.canonicalize_dtype(_dtype(x), allow_opaque_dtype=True),
                jax.dtypes.canonicalize_dtype(_dtype(y), allow_opaque_dtype=True),
            )
        else:
            self.assertEqual(_dtype(x), _dtype(y))


class PrimitivesTest(JaxRunningTestCase):
    @primitives_parameterized(
        primitive_harness.all_harnesses,
    )
    @ignore_warning(
        category=UserWarning, message="Using reduced precision for gradient.*"
    )
    def test_prim(self, harness: primitive_harness.Harness):
        def _filter_limitation(limitation):
            return limitation.filter(device="cpu", dtype=harness.dtype, mode="compiled")

        limitations = tuple(
            filter(
                _filter_limitation,
                jax2tf_limitations.Jax2TfLimitation.limitations_for_harness(harness),
            )
        )

        devices = []
        possible_backends = ["iree_cpu"]
        for backend in possible_backends:
            try:
                devices.extend(jax.devices(backend))
            except RuntimeError:
                continue

        func_jax = harness.dyn_fun
        args = harness.dyn_args_maker(self.rng())

        baseline_platform: Optional[str] = None
        baseline_results = None
        for d in devices:
            if baseline_platform is None or baseline_platform != d.platform:
                device_args = jax.tree_util.tree_map(
                    lambda x: jax.device_put(x, d), args
                )
                logging.info("Running harness on %s", d)
                with jax.jax2tf_associative_scan_reductions(True):
                    res = func_jax(*device_args)
                if baseline_platform is None:
                    baseline_platform = d.platform
                    baseline_results = res
                else:
                    logging.info(
                        "Comparing results for %s and %s", baseline_platform, d.platform
                    )
                    self.assert_all_close(baseline_results, res)


if __name__ == "__main__":
    absltest.main()
