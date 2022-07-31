# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import gc
import unittest

import iree.runtime as rt

NONE_CTOR = lambda iface: None


class PyModuleInterfaceTest(unittest.TestCase):

  def setUp(self):
    self._instance = rt.VmInstance()

  def testEmptyModuleLifecycle(self):
    iface = rt.PyModuleInterface("test1", NONE_CTOR)
    print(iface)
    self.assertFalse(iface.initialized)
    m = iface.create()
    print(iface)
    self.assertTrue(iface.initialized)
    print(m)
    m = None
    gc.collect()
    print(iface)
    self.assertTrue(iface.destroyed)

  def testEmptyModuleInstance(self):
    iface = rt.PyModuleInterface("test1", NONE_CTOR)
    m = iface.create()
    context = rt.VmContext(self._instance, modules=(m,))
    self.assertTrue(iface.initialized)
    print(context)

    # Make sure no circular refs and that everything frees.
    context = None
    m = None
    gc.collect()
    self.assertTrue(iface.destroyed)

  def testMultiModuleInstance(self):
    calls = []

    def ctor(iface):
      calls.append(iface)
      return None

    iface = rt.PyModuleInterface("test1", ctor)
    m = iface.create()
    context1 = rt.VmContext(self._instance, modules=(m,))
    self.assertTrue(iface.initialized)
    context2 = rt.VmContext(self._instance, modules=(m,))
    self.assertTrue(iface.initialized)
    self.assertEqual(2, len(calls))

    # Make sure no circular refs and that everything frees.
    calls = None
    context1 = None
    m = None
    context2 = None
    gc.collect()
    self.assertTrue(iface.destroyed)

  def testVoidFunctionExport(self):
    messages = []

    class Methods:

      def __init__(self, iface):
        self.iface = iface
        self.counter = 0

      def say_hello(self):
        messages.append(f"Hello! Your number is {self.counter}")
        print(messages[-1])
        self.counter += 1

    iface = rt.PyModuleInterface("test1", Methods)
    iface.export("say_hello", "0v", Methods.say_hello)
    m = iface.create()
    context = rt.VmContext(self._instance, modules=(m,))
    f = m.lookup_function("say_hello")
    self.assertIsNotNone(f)
    args = rt.VmVariantList(0)
    results = rt.VmVariantList(0)

    # Invoke twice - should produce two messages.
    context.invoke(f, args, results)
    context.invoke(f, args, results)
    self.assertListEqual(messages, [
        "Hello! Your number is 0",
        "Hello! Your number is 1",
    ])

    # Make sure no circular refs and that everything frees.
    context = None
    m = None
    gc.collect()
    self.assertTrue(iface.destroyed)

  def testPythonException(self):
    messages = []

    class Methods:

      def __init__(self, iface):
        pass

      def do_it(self):
        raise ValueError("This is from Python")

    iface = rt.PyModuleInterface("test1", Methods)
    iface.export("do_it", "0v", Methods.do_it)
    m = iface.create()
    context = rt.VmContext(self._instance, modules=(m,))
    f = m.lookup_function("do_it")
    self.assertIsNotNone(f)
    args = rt.VmVariantList(0)
    results = rt.VmVariantList(0)

    # We are testing here that the Python level exception is caught and
    # translated to an IREE status (surfacing as a RuntimeError) vs percolating
    # through the C call stack.
    with self.assertRaisesRegex(RuntimeError,
                                "ValueError: This is from Python"):
      context.invoke(f, args, results)

    # Make sure no circular refs and that everything frees.
    context = None
    m = None
    gc.collect()
    self.assertTrue(iface.destroyed)

  def testPrimitiveArguments(self):
    values = []

    class Methods:

      def __init__(self, iface):
        pass

      def do_it(self, a, b):
        values.append((a, b))

    iface = rt.PyModuleInterface("test1", Methods)
    iface.export("do_it_i32", "0ii", Methods.do_it)
    iface.export("do_it_i64", "0II", Methods.do_it)
    iface.export("do_it_f32", "0ff", Methods.do_it)
    iface.export("do_it_f64", "0FF", Methods.do_it)
    m = iface.create()
    context = rt.VmContext(self._instance, modules=(m,))

    args = rt.VmVariantList(2)
    results = rt.VmVariantList(0)
    args.push_int(42)
    args.push_int(43)
    context.invoke(m.lookup_function("do_it_i32"), args, results)
    context.invoke(m.lookup_function("do_it_i64"), args, results)

    args = rt.VmVariantList(2)
    args.push_float(2.0)
    args.push_float(4.0)
    # TODO: Python doesn't have 32bit floats, so we are populating f64 args.
    # These are coming back as zeros, and I expected something to be
    # doing a conversion? The same is being done with i64 above but is
    # working there.
    context.invoke(m.lookup_function("do_it_f32"), args, results)
    context.invoke(m.lookup_function("do_it_f64"), args, results)

    print(values)
    self.assertEqual(repr(values),
                     "[(42, 43), (42, 43), (0.0, 0.0), (2.0, 4.0)]")

    # Make sure no circular refs and that everything frees.
    context = None
    m = None
    gc.collect()
    self.assertTrue(iface.destroyed)

  def testPrimitiveResults(self):
    next_results = None

    class Methods:

      def __init__(self, iface):
        pass

      def do_it(self):
        return next_results

    iface = rt.PyModuleInterface("test1", Methods)
    iface.export("do_it_i32", "0v_ii", Methods.do_it)
    iface.export("do_it_i64", "0v_II", Methods.do_it)
    iface.export("do_it_f32", "0v_ff", Methods.do_it)
    iface.export("do_it_f64", "0v_FF", Methods.do_it)
    iface.export("do_it_unary_i32", "0v_i", Methods.do_it)
    m = iface.create()
    context = rt.VmContext(self._instance, modules=(m,))

    args = rt.VmVariantList(0)

    # i32
    results = rt.VmVariantList(2)
    next_results = (42, 43)
    context.invoke(m.lookup_function("do_it_i32"), args, results)
    self.assertEqual(repr(results), "<VmVariantList(2): [42, 43]>")

    # i64
    results = rt.VmVariantList(2)
    next_results = (42, 43)
    context.invoke(m.lookup_function("do_it_i64"), args, results)
    self.assertEqual(repr(results), "<VmVariantList(2): [42, 43]>")

    # f32
    results = rt.VmVariantList(2)
    next_results = (2.0, 4.0)
    context.invoke(m.lookup_function("do_it_f32"), args, results)
    self.assertEqual(repr(results), "<VmVariantList(2): [2.000000, 4.000000]>")

    # f64
    results = rt.VmVariantList(2)
    next_results = (2.0, 4.0)
    context.invoke(m.lookup_function("do_it_f64"), args, results)
    self.assertEqual(repr(results), "<VmVariantList(2): [2.000000, 4.000000]>")

    # Unary special case.
    results = rt.VmVariantList(1)
    next_results = (42)
    context.invoke(m.lookup_function("do_it_unary_i32"), args, results)
    self.assertEqual(repr(results), "<VmVariantList(1): [42]>")

    # Make sure no circular refs and that everything frees.
    context = None
    m = None
    gc.collect()
    self.assertTrue(iface.destroyed)

  def testRefArguments(self):
    values = []

    class Methods:

      def __init__(self, iface):
        pass

      def do_it(self, a, b):
        values.append((a.deref(rt.VmVariantList), b.deref(rt.VmVariantList)))

    iface = rt.PyModuleInterface("test1", Methods)
    iface.export("do_it", "0rr", Methods.do_it)
    m = iface.create()
    context = rt.VmContext(self._instance, modules=(m,))

    # These lists just happen to be reference objects we know how to
    # create.
    arg0 = rt.VmVariantList(1)
    arg0.push_int(42)
    arg1 = rt.VmVariantList(1)
    arg1.push_int(84)

    args = rt.VmVariantList(2)
    args.push_list(arg0)
    args.push_list(arg1)
    results = rt.VmVariantList(2)
    context.invoke(m.lookup_function("do_it"), args, results)
    print("REF VALUES:", values)
    self.assertEqual(repr(values),
                     "[(<VmVariantList(1): [42]>, <VmVariantList(1): [84]>)]")

  def testRefResults(self):

    class Methods:

      def __init__(self, iface):
        pass

      def do_it(self):
        # These lists just happen to be reference objects we know how to
        # create.
        r0 = rt.VmVariantList(1)
        r0.push_int(42)
        r1 = rt.VmVariantList(1)
        r1.push_int(84)
        return r0.ref, r1.ref

    iface = rt.PyModuleInterface("test1", Methods)
    iface.export("do_it", "0v_rr", Methods.do_it)
    m = iface.create()
    context = rt.VmContext(self._instance, modules=(m,))

    args = rt.VmVariantList(0)
    results = rt.VmVariantList(2)
    context.invoke(m.lookup_function("do_it"), args, results)
    print("REF RESULTS:", results)
    self.assertEqual(repr(results), "<VmVariantList(2): [List[42], List[84]]>")


if __name__ == "__main__":
  unittest.main()
