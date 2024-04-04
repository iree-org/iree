# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import array
import logging
import numpy as np
from pathlib import Path
import tempfile
import unittest

import iree.runtime as rt


def _float_constant(val: float) -> array.array:
    return array.array("f", [val])


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
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "archive.irpa"
            orig_array = np.asarray([[1], [2], [3]], dtype=np.int64)
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

            # Load and verify.
            index = rt.ParameterIndex()
            index.load(str(file_path))
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
