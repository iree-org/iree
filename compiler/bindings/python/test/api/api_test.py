# Copyright 2023 Stella Laurenzo
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# TODO: Upstream this to IREE.

from contextlib import closing
from pathlib import Path
import tempfile
import os
import unittest

from iree.compiler.api import *
from iree.compiler import ir


class DlFlagsTest(unittest.TestCase):
    def testDefaultFlags(self):
        session = Session()
        flags = session.get_flags()
        print(flags)
        self.assertIn("--iree-input-type=auto", flags)

    def testNonDefaultFlags(self):
        session = Session()
        flags = session.get_flags(non_default_only=True)
        self.assertEqual(flags, [])
        session.set_flags("--iree-input-type=none")
        flags = session.get_flags(non_default_only=True)
        self.assertIn("--iree-input-type=none", flags)

    def testFlagsAreScopedToSession(self):
        session1 = Session()
        session2 = Session()
        session1.set_flags("--iree-input-type=tosa")
        session2.set_flags("--iree-input-type=none")
        self.assertIn("--iree-input-type=tosa", session1.get_flags())
        self.assertIn("--iree-input-type=none", session2.get_flags())

    def testFlagError(self):
        session = Session()
        with self.assertRaises(ValueError):
            session.set_flags("--does-not-exist=1")


class DlInvocationTest(unittest.TestCase):
    def testCreate(self):
        session = Session()
        inv = session.invocation()


class DlOutputTest(unittest.TestCase):
    def testOpenMembuffer(self):
        out = Output.open_membuffer()

    def testOpenMembufferExplicitClose(self):
        out = Output.open_membuffer()
        out.close()

    def testOpenMembufferWrite(self):
        out = Output.open_membuffer()
        out.write(b"foobar")
        mem = out.map_memory()
        self.assertEqual(b"foobar", bytes(mem))
        out.close()

    def testOpenFileNoKeep(self):
        file_path = tempfile.mktemp()
        out = Output.open_file(file_path)
        try:
            out.write(b"foobar")
            self.assertTrue(Path(file_path).exists())
        finally:
            out.close()
            # Didn't call keep, so should be deleted.
        self.assertFalse(Path(file_path).exists())

    def testOpenFileKeep(self):
        file_path = tempfile.mktemp()
        out = Output.open_file(file_path)
        try:
            try:
                out.write(b"foobar")
                out.keep()
            finally:
                out.close()
                # Didn't call keep, so should be deleted.

            with open(file_path, "rb") as f:
                contents = f.read()
                self.assertEqual(b"foobar", contents)
        finally:
            Path(file_path).unlink()


class DlInteropTest(unittest.TestCase):
    def testContextFromSession(self):
        s = Session()
        # TODO: Test that multiple calls return the same context.
        # TODO: Do gc stuff to verify memory.
        context1 = s.context
        context2 = s.context
        self.assertIsNotNone(context1)
        self.assertIs(context1, context2)

    def testImportModule(self):
        s = Session()
        with ir.Location.unknown(s.context):
            module_op = ir.Module.create().operation
            module_op.attributes["test.test"] = ir.Attribute.parse('"working"')
            inv = s.invocation()
            inv.import_module(module_op)
            # Round-trip it back through an Output and verify that the attribute
            # we set is still there.
            output = Output.open_membuffer()
            inv.output_ir(output)
            contents = bytes(output.map_memory()).decode()
            print(contents)
            self.assertIn('test.test = "working"', contents)


# TODO: Port these to the current API.
# class CompilerAPITest(unittest.TestCase):
#     def testCreate(self):
#         compiler = Compiler()

#     def testLoadFromBytes(self):
#         compiler = Compiler()
#         p = compiler.load_buffer("module {}".encode(), buffer_name="foobar")

#     def testPipelineClose(self):
#         compiler = Compiler()
#         p = compiler.load_buffer("module {}".encode(), buffer_name="foobar")
#         p.close()

#     def testLoadFromFile(self):
#         compiler = Compiler()
#         with tempfile.NamedTemporaryFile("w", delete=False) as tf:
#             tf.write("module {}")
#             tf.close()
#             p = compiler.load_file(tf.name)
#             p.close()

#     def testExecuteIR(self):
#         compiler = Compiler()
#         p = compiler.load_buffer("module {}".encode(), buffer_name="foobar")
#         p.execute()
#         with closing(compiler.open_output_membuffer()) as output:
#             p.output_ir(output)
#             ir_contents = bytes(output.map_memory())
#             print(ir_contents)
#             self.assertEqual(b"module {\n}", ir_contents)

#     def testExecuteVMFB(self):
#         compiler = Compiler()
#         compiler.set_flags("--iree-hal-target-backends=vmvx")
#         p = compiler.load_buffer(
#             "module {func.func @main(%arg0: i32) -> (i32) {return %arg0 : i32}}".encode(),
#             buffer_name="foobar",
#         )
#         p.execute()
#         with closing(compiler.open_output_membuffer()) as output:
#             p.output_vm_bytecode(output)
#             ir_contents = bytes(output.map_memory())
#             print(len(ir_contents))
#             self.assertGreater(len(ir_contents), 0)


if __name__ == "__main__":
    if os.system() == "Windows":
        print("WARNING: Test disabled on Windows due to suspected MSVC bug")
    else:
        unittest.main()
