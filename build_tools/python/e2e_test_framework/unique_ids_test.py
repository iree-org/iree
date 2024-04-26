# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import hashlib
import unittest

import unique_ids


class UniqueIdsTest(unittest.TestCase):
    def test_hash_composite_id(self):
        output = unique_ids.hash_composite_id(["abc", "123"])

        self.assertEqual(
            output, hashlib.sha256(f"0-abc:1-123".encode("utf-8")).hexdigest()
        )

    def test_hash_composite_id_diff_keys(self):
        ids = [
            unique_ids.hash_composite_id([]),
            unique_ids.hash_composite_id(["abc", "123"]),
            unique_ids.hash_composite_id(["123", "abc"]),
            unique_ids.hash_composite_id(["123", unique_ids.TRANSPARENT_ID]),
            unique_ids.hash_composite_id(["123", "abc", "xyz"]),
            unique_ids.hash_composite_id(["123", unique_ids.TRANSPARENT_ID, "xyz"]),
        ]

        # Check if they are all distinct.
        self.assertCountEqual(set(ids), ids)

    def test_hash_composite_id_unchanged_with_transparent_id(self):
        existing_id = unique_ids.hash_composite_id(["abc"])
        new_id_a = unique_ids.hash_composite_id(["abc", unique_ids.TRANSPARENT_ID])
        new_id_b = unique_ids.hash_composite_id(
            ["abc", unique_ids.TRANSPARENT_ID, unique_ids.TRANSPARENT_ID]
        )

        self.assertEqual(existing_id, new_id_a)
        self.assertEqual(existing_id, new_id_b)

    def test_hash_composite_id_with_transparent_ids_in_diff_pos(self):
        id_a = unique_ids.hash_composite_id([unique_ids.TRANSPARENT_ID, "abc"])
        id_b = unique_ids.hash_composite_id(["abc", unique_ids.TRANSPARENT_ID])

        self.assertNotEqual(id_a, id_b)


if __name__ == "__main__":
    unittest.main()
