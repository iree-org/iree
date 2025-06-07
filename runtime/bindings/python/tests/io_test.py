# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import array
import gc
import logging
import numpy as np
from pathlib import Path
import tempfile
import unittest

import iree.runtime as rt


def _float_constant(val: float) -> array.array:
    return array.array("f", [val])


def index_entry_as_array(entry, like_array):
    return np.asarray(entry.file_view).view(like_array.dtype).reshape(like_array.shape)


class ParameterApiTest(unittest.TestCase):
    def testCreateArchiveFile(self):
        splat_index = rt.ParameterIndex()
        splat_index.add_splat("weight", _float_constant(2.0), 30 * 20 * 4)
        splat_index.add_splat("bias", _float_constant(1.0), 30 * 4)

        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "archive.irpa"
            target_index = splat_index.create_archive_file(str(file_path))
            print(target_index)
            self.assertTrue(file_path.exists())
            self.assertGreater(file_path.stat().st_size, 0)

    def testArchiveFileRoundtrip(self):
        orig_array = np.asarray([[1], [2], [3]], dtype=np.int64)

        def verify_archive(file_path: Path):
            # Load and verify.
            index = rt.ParameterIndex()
            # For this test, disable mmap to make temp file management on
            # windows a bit better.
            index.load(str(file_path), mmap=False)
            self.assertEqual(len(index), 3)

            # Note that the happy path of most properties are verified via
            # the repr (as they are called internal to that).
            entries = dict(index.items())
            self.assertEqual(
                repr(entries["weight"]),
                "<ParameterIndexEntry 'weight' splat b'\\x02':600>",
            )
            self.assertEqual(
                repr(entries["bias"]),
                "<ParameterIndexEntry 'bias' splat b' ':30>",
            )
            self.assertRegex(
                repr(entries["array"]),
                r"<ParameterIndexEntry 'array' FileHandle<host_allocation\(.*\)>:384:24",
            )

            # Verify some non-happy paths.
            with self.assertRaisesRegex(ValueError, "Entry is not file storage based"):
                entries["weight"].file_storage
            with self.assertRaisesRegex(ValueError, "Entry is not splat"):
                entries["array"].splat_pattern

            # Verify that the repr of the index itself is sensical.
            index_repr = repr(index)
            self.assertIn("Parameter scope <global> (3 entries", index_repr)

            # Get the array contents and verify against original.
            array_view = entries["array"].file_view
            self.assertEqual(len(array_view), 24)
            array_back = np.asarray(array_view).view(np.int64).reshape(orig_array.shape)
            np.testing.assert_array_equal(array_back, orig_array)

        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "archive.irpa"
            rt.save_archive_file(
                {
                    "weight": rt.SplatValue(np.int8(2), [30, 20]),
                    "bias": rt.SplatValue(array.array("b", [32]), 30),
                    "array": orig_array,
                },
                file_path,
            )
            self.assertTrue(file_path.exists())
            self.assertGreater(file_path.stat().st_size, 0)
            # Open / verify in its own scope and collect prior to tearing
            # down the temp dir.
            verify_archive(file_path)
            gc.collect()

    def testArchiveFileRoundtripByFD(self):
        orig_array = np.asarray([[1], [2], [3]], dtype=np.int64)

        def verify_archive_from_fd(file_path: Path):
            f = open(file_path, "rb")
            self.assertIsNotNone(f)

            handle = rt.FileHandle.wrap_fd(f.fileno(), True, False)

            # Load and verify.
            index = rt.ParameterIndex()
            # For this test, disable mmap to make temp file management on
            # windows a bit better.
            index.load_from_file_handle(handle, "irpa")
            self.assertEqual(len(index), 3)

            # Note that the happy path of most properties are verified via
            # the repr (as they are called internal to that).
            entries = dict(index.items())
            self.assertEqual(
                repr(entries["weight"]),
                "<ParameterIndexEntry 'weight' splat b'\\x02':600>",
            )
            self.assertEqual(
                repr(entries["bias"]),
                "<ParameterIndexEntry 'bias' splat b' ':30>",
            )
            self.assertRegex(
                repr(entries["array"]),
                r"<ParameterIndexEntry 'array' FileHandle<fd\(.*\)>",
            )

            # Verify some non-happy paths.
            with self.assertRaisesRegex(ValueError, "Entry is not file storage based"):
                entries["weight"].file_storage
            with self.assertRaisesRegex(ValueError, "Entry is not splat"):
                entries["array"].splat_pattern

            # Verify that the repr of the index itself is sensical.
            index_repr = repr(index)
            self.assertIn("Parameter scope <global> (3 entries", index_repr)

            f.close()

        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "archive.irpa"
            rt.save_archive_file(
                {
                    "weight": rt.SplatValue(np.int8(2), [30, 20]),
                    "bias": rt.SplatValue(array.array("b", [32]), 30),
                    "array": orig_array,
                },
                file_path,
            )
            self.assertTrue(file_path.exists())
            self.assertGreater(file_path.stat().st_size, 0)
            # Open / verify in its own scope and collect prior to tearing
            # down the temp dir.
            verify_archive_from_fd(file_path)
            gc.collect()

    def testParameterIndexEntryFromToNumpy(self):
        array = np.array([[1, 2], [3, 4]], dtype=np.int32)
        index = rt.ParameterIndex()
        key = "key"
        rt.parameter_index_add_numpy_ndarray(index, key, array)
        assert index.items()[0][0] == key
        index_entry_as_array = rt.parameter_index_entry_as_numpy_ndarray(
            index.items()[0][1]
        )
        np.testing.assert_equal(index_entry_as_array, array)

    def testParameterIndexEntryFromToNumpyZeroDims(self):
        array = np.array(1234, dtype=np.int32)
        index = rt.ParameterIndex()
        key = "key"
        rt.parameter_index_add_numpy_ndarray(index, key, array)
        assert index.items()[0][0] == key
        index_entry_as_array = rt.parameter_index_entry_as_numpy_ndarray(
            index.items()[0][1]
        )
        np.testing.assert_equal(index_entry_as_array, array)

    def testParameterIndexEntryFromIreeTurbine(self):
        """Verify that we are able to load a tensor from IRPA generated with IREE
        Turbine.
        We want to maintain backward compatibility with existing IRPA files."""
        index = rt.ParameterIndex()
        irpa_path = str(
            Path(__file__).resolve().parent
            / "testdata"
            / "tensor_saved_with_iree_turbine.irpa"
        )
        index.load(irpa_path)
        items = index.items()
        assert len(items) == 1
        key, entry = items[0]
        assert key == "the_torch_tensor"
        index_entry_as_array = rt.parameter_index_entry_as_numpy_ndarray(entry)
        expected_array = np.array([1, 2, 3, 4], dtype=np.uint8)
        np.testing.assert_array_equal(index_entry_as_array, expected_array, strict=True)

    def testFileHandleWrap(self):
        fh = rt.FileHandle.wrap_memory(b"foobar")
        view = fh.host_allocation
        del fh
        self.assertEqual(bytes(view), b"foobar")

    def testFileHandleWrap(self):
        fh = rt.FileHandle.wrap_memory(b"foobar")
        view = fh.host_allocation
        del fh
        self.assertEqual(bytes(view), b"foobar")

    def testParameterIndexAddFromFile(self):
        splat_index = rt.ParameterIndex()
        fh = rt.FileHandle.wrap_memory(b"foobar")
        splat_index.add_from_file_handle("data", fh, length=3, offset=3)

    def testSplatTooBig(self):
        splat_index = rt.ParameterIndex()
        with self.assertRaises(ValueError):
            splat_index.add_splat(
                "weight", array.array("f", [1.0, 2.0, 3.0, 4.0, 5.0]), 30 * 20 * 4
            )

    def testGguf(self):
        index = rt.ParameterIndex()
        index.load(
            str(
                Path(__file__).resolve().parent
                / "testdata"
                / "parameter_weight_bias_1.gguf"
            )
        )
        expected_weight = np.zeros([30, 20], dtype=np.float32) + 2.0
        expected_bias = np.zeros([30], dtype=np.float32) + 1.0
        entries = dict(index.items())
        weight = index_entry_as_array(entries["weight"], expected_weight)
        bias = index_entry_as_array(entries["bias"], expected_bias)
        np.testing.assert_array_equal(weight, expected_weight)
        np.testing.assert_array_equal(bias, expected_bias)

    def testSafetensors(self):
        index = rt.ParameterIndex()
        index.load(
            str(
                Path(__file__).resolve().parent
                / "testdata"
                / "parameter_weight_bias_1.safetensors"
            )
        )
        expected_weight = np.zeros([30, 20], dtype=np.float32) + 2.0
        expected_bias = np.zeros([30], dtype=np.float32) + 1.0
        entries = dict(index.items())
        weight = index_entry_as_array(entries["weight"], expected_weight)
        bias = index_entry_as_array(entries["bias"], expected_bias)
        np.testing.assert_array_equal(weight, expected_weight)
        np.testing.assert_array_equal(bias, expected_bias)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
