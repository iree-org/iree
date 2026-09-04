#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tests for gen_rename_map."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import gen_rename_map

COMPONENT = gen_rename_map._itanium_component("IREE18")


class ItaniumComponentTest(unittest.TestCase):
    def test_component_is_length_prefixed(self):
        self.assertEqual(COMPONENT, "6IREE18")

    def test_length_digit_follows_the_prefix(self):
        # A fork setting its own prefix gets the right length digit, so the
        # prefix is never spelled out with the digit attached.
        self.assertEqual(gen_rename_map._itanium_component("AB"), "2AB")
        self.assertEqual(
            gen_rename_map._itanium_component("LONGERPREFIX"), "12LONGERPREFIX"
        )

    def test_rename_uses_the_given_component(self):
        self.assertEqual(
            gen_rename_map._renamed(
                "_ZN4mlir3fooEv", "mlir::foo()", gen_rename_map._itanium_component("AB")
            ),
            "_ZN2AB4mlir3fooEv",
        )


class NeedsRenameTest(unittest.TestCase):
    def test_namespace_qualifiers_match(self):
        self.assertTrue(gen_rename_map._needs_rename("mlir::func::FuncOp::print()"))
        self.assertTrue(gen_rename_map._needs_rename("foo(mlir::Value)"))
        self.assertTrue(gen_rename_map._needs_rename("vtable for llvm::raw_ostream"))
        self.assertTrue(
            gen_rename_map._needs_rename(
                "std::vector<mlir::Value>::push_back(mlir::Value&&)"
            )
        )

    def test_review_false_positive_good4mlir(self):
        # `_Z9good4mlirv` has the raw `4mlir` marker but no mlir:: namespace.
        self.assertFalse(gen_rename_map._needs_rename("good4mlir()"))

    def test_namespace_suffix_does_not_match(self):
        self.assertFalse(gen_rename_map._needs_rename("myllvm::foo()"))
        self.assertFalse(gen_rename_map._needs_rename("notmlir::bar()"))

    def test_unrelated_names_do_not_match(self):
        self.assertFalse(gen_rename_map._needs_rename("iree_compiler::foo()"))
        self.assertFalse(gen_rename_map._needs_rename("plain_c_symbol"))


class RenamedTest(unittest.TestCase):
    def test_nested_name(self):
        self.assertEqual(
            gen_rename_map._renamed("_ZN4mlir3fooEv", "mlir::foo()", COMPONENT),
            "_ZN6IREE184mlir3fooEv",
        )

    def test_macho_extra_underscore(self):
        self.assertEqual(
            gen_rename_map._renamed("__ZN4mlir3barEv", "mlir::bar()", COMPONENT),
            "__ZN6IREE184mlir3barEv",
        )

    def test_vtable_typeinfo_guard(self):
        self.assertEqual(
            gen_rename_map._renamed(
                "_ZTVN4mlir6WalkerE", "vtable for mlir::Walker", COMPONENT
            ),
            "_ZTVN6IREE184mlir6WalkerE",
        )
        self.assertEqual(
            gen_rename_map._renamed(
                "_ZTIN4llvm5ErrorE", "typeinfo for llvm::Error", COMPONENT
            ),
            "_ZTIN6IREE184llvm5ErrorE",
        )
        self.assertEqual(
            gen_rename_map._renamed(
                "_ZGVN4mlir3ctxE", "guard variable for mlir::ctx", COMPONENT
            ),
            "_ZGVN6IREE184mlir3ctxE",
        )

    def test_local_scope_and_guard_chains(self):
        # Guard variable of a function-local static (the MLIR TypeID pattern).
        self.assertEqual(
            gen_rename_map._renamed(
                "__ZGVZN4mlir6detail14TypeIDResolverIN4llvm5APIntEvE13resolveTypeIDEvE2id",
                "guard variable for mlir::detail::TypeIDResolver<llvm::APInt, "
                "void>::resolveTypeID()::id",
                COMPONENT,
            ),
            "__ZGVZN6IREE184mlir6detail14TypeIDResolverIN4llvm5APIntEvE13resolveTypeIDEvE2id",
        )
        # The local static itself.
        self.assertEqual(
            gen_rename_map._renamed(
                "_ZZN4mlir3fooEvE5local",
                "mlir::foo()::local",
                COMPONENT,
            ),
            "_ZZN6IREE184mlir3fooEvE5local",
        )

    def test_free_function_with_mlir_argument(self):
        # A plugin function naming MLIR types must agree with its call sites.
        self.assertEqual(
            gen_rename_map._renamed(
                "_Z3fooN4mlir5ValueE", "foo(mlir::Value)", COMPONENT
            ),
            "_Z6IREE183fooN4mlir5ValueE",
        )

    def test_review_false_positive_good4mlir(self):
        self.assertIsNone(
            gen_rename_map._renamed("_Z9good4mlirv", "good4mlir()", COMPONENT)
        )

    def test_non_itanium_symbols_pass_through(self):
        self.assertIsNone(
            gen_rename_map._renamed("plain_c_symbol", "plain_c_symbol", COMPONENT)
        )
        self.assertIsNone(
            gen_rename_map._renamed("mlirContextCreate", "mlirContextCreate", COMPONENT)
        )

    def test_iree_namespace_not_renamed(self):
        self.assertIsNone(
            gen_rename_map._renamed("_ZN4iree3bazEv", "iree::baz()", COMPONENT)
        )


class WriteRenameMapTest(unittest.TestCase):
    def test_writes_sorted_map(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = os.path.join(temp_dir, "rename.map")
            gen_rename_map._write_rename_map(
                {
                    ("_ZN4mlir3barEv", "_ZN6IREE184mlir3barEv"),
                    ("_ZN4llvm3fooEv", "_ZN6IREE184llvm3fooEv"),
                },
                out_path,
            )
            with open(out_path, encoding="utf-8") as f:
                self.assertEqual(
                    f.read(),
                    "_ZN4llvm3fooEv _ZN6IREE184llvm3fooEv\n"
                    "_ZN4mlir3barEv _ZN6IREE184mlir3barEv\n",
                )


if __name__ == "__main__":
    unittest.main()
