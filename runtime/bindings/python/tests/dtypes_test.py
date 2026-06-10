# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import unittest

from iree import runtime as rt
import iree.runtime.dtypes as dtypes
import ml_dtypes


class DTypesTest(unittest.TestCase):
    def testDTypeToInfo(self):
        info = dtypes.map_dtype_to_dtype_info(np.dtype(np.bool_))
        self.assertEqual(info.dtype, np.dtype(np.bool_))
        self.assertEqual(info.name, "bool")
        self.assertEqual(info.abi_type, "i1")
        self.assertFalse(info.is_integer)
        self.assertFalse(info.is_signed)
        self.assertFalse(info.is_floating_point)
        self.assertFalse(info.is_complex)

        with self.assertRaises(KeyError):
            dtypes.map_dtype_to_dtype_info(np.dtype("bfloat16"))

    def testNameToInfo(self):
        info = dtypes.map_name_to_dtype_info("float32")
        self.assertEqual(info.dtype, np.dtype(np.float32))
        self.assertEqual(info.name, "float32")
        self.assertEqual(info.abi_type, "f32")
        self.assertFalse(info.is_integer)
        self.assertFalse(info.is_signed)
        self.assertTrue(info.is_floating_point)
        self.assertFalse(info.is_complex)

        with self.assertRaises(KeyError):
            dtypes.map_name_to_dtype_info("invalid")

    def testAbiTypeToInfo(self):
        info = dtypes.map_abi_type_to_dtype_info("i16")
        self.assertEqual(info.dtype, np.dtype(np.int16))
        self.assertEqual(info.name, "int16")
        self.assertEqual(info.abi_type, "i16")
        self.assertTrue(info.is_integer)
        self.assertTrue(info.is_signed)
        self.assertFalse(info.is_floating_point)
        self.assertFalse(info.is_complex)

        with self.assertRaises(KeyError):
            dtypes.map_abi_type_to_dtype_info("invalid")

        self.assertEqual(
            dtypes.map_abi_type_to_dtype_info("i16").dtype, np.dtype(np.int16)
        )
        self.assertEqual(
            dtypes.map_abi_type_to_dtype_info("i32").dtype, np.dtype(np.int32)
        )
        self.assertEqual(
            dtypes.map_abi_type_to_dtype_info("i64").dtype, np.dtype(np.int64)
        )

    def testDTypeToAbiType(self) -> None:
        self.assertEqual(dtypes.DTYPE_TO_ABI_TYPE[np.dtype(np.uint8)], "i8")
        self.assertEqual(dtypes.DTYPE_TO_ABI_TYPE[np.dtype(np.uint16)], "i16")
        self.assertEqual(dtypes.DTYPE_TO_ABI_TYPE[np.dtype(np.uint32)], "i32")

    def testMapMlDtypeToHalElementTypeFromNumpyDType(self):
        self.assertEqual(
            dtypes.map_dtype_to_hal_element_type(np.dtype("bfloat16")),
            rt.HalElementType.BFLOAT_16,
        )
        self.assertEqual(
            dtypes.map_dtype_to_hal_element_type(np.dtype("float8_e4m3fn")),
            rt.HalElementType.FLOAT_8_E4M3_FN,
        )
        self.assertEqual(
            dtypes.map_dtype_to_hal_element_type(np.dtype("float8_e4m3fnuz")),
            rt.HalElementType.FLOAT_8_E4M3_FNUZ,
        )
        self.assertEqual(
            dtypes.map_dtype_to_hal_element_type(np.dtype("float8_e5m2")),
            rt.HalElementType.FLOAT_8_E5M2,
        )
        self.assertEqual(
            dtypes.map_dtype_to_hal_element_type(np.dtype("float8_e5m2fnuz")),
            rt.HalElementType.FLOAT_8_E5M2_FNUZ,
        )
        self.assertEqual(
            dtypes.map_dtype_to_hal_element_type(np.dtype("float8_e8m0fnu")),
            rt.HalElementType.FLOAT_8_E8M0_FNU,
        )

    def testMapMlDtypeToHalElementTypeFromScalarType(self):
        self.assertEqual(
            dtypes.map_dtype_to_hal_element_type(ml_dtypes.bfloat16),
            rt.HalElementType.BFLOAT_16,
        )
        self.assertEqual(
            dtypes.map_dtype_to_hal_element_type(ml_dtypes.float8_e4m3fn),
            rt.HalElementType.FLOAT_8_E4M3_FN,
        )


if __name__ == "__main__":
    unittest.main()
