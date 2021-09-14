# Lint as: python3
# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import numpy as np

from absl.testing import absltest

from iree import runtime as rt
from iree.runtime.function import FunctionInvoker
from iree.runtime.binding import VmVariantList


class MockVmContext:

  def __init__(self, invoke_callback):
    self._invoke_callback = invoke_callback
    self.invocations = []

  def invoke(self, vm_function, arg_list, ret_list):
    self._invoke_callback(arg_list, ret_list)
    self.invocations.append((vm_function, arg_list, ret_list))
    print(f"INVOKE: {arg_list} -> {ret_list}")

  @property
  def mock_arg_reprs(self):
    return repr([arg_list for _, arg_list, _ in self.invocations])


class MockVmFunction:

  def __init__(self, reflection):
    self.reflection = reflection


class FunctionTest(absltest.TestCase):

  def setUp(self):
    # Doesn't matter what device. We just need one.
    config = rt.Config("vmvx")
    self.device = config.device

  def testNoReflectionScalars(self):

    def invoke(arg_list, ret_list):
      ret_list.push_int(3)
      ret_list.push_int(4)

    vm_context = MockVmContext(invoke)
    vm_function = MockVmFunction(reflection={})
    invoker = FunctionInvoker(vm_context, self.device, vm_function, tracer=None)
    result = invoker(1, 2)
    self.assertEqual("[<VmVariantList(2): [1, 2]>]", vm_context.mock_arg_reprs)
    self.assertEqual((3, 4), result)

  def testKeywordArgs(self):

    def invoke(arg_list, ret_list):
      ret_list.push_int(3)

    vm_context = MockVmContext(invoke)
    vm_function = MockVmFunction(
        reflection={
            "iree.abi":
                json.dumps({
                    "a": [
                        "i32",
                        ["named", "a", "i32"],
                        ["named", "b", "i32"],
                    ],
                    "r": ["i32",],
                })
        })
    invoker = FunctionInvoker(vm_context, self.device, vm_function, tracer=None)
    result = invoker(-1, a=1, b=2)
    self.assertEqual("[<VmVariantList(3): [-1, 1, 2]>]",
                     vm_context.mock_arg_reprs)
    self.assertEqual(3, result)

  def testInlinedResults(self):

    def invoke(arg_list, ret_list):
      ret_list.push_int(3)
      ret_list.push_int(4)

    vm_context = MockVmContext(invoke)
    vm_function = MockVmFunction(reflection={
        "iree.abi": json.dumps({
            "a": [],
            "r": [["slist", "i32", "i32"]],
        })
    })
    invoker = FunctionInvoker(vm_context, self.device, vm_function, tracer=None)
    result = invoker()
    self.assertEqual([3, 4], result)

  def testNestedResults(self):

    def invoke(arg_list, ret_list):
      ret_list.push_int(3)
      sub_list = VmVariantList(2)
      sub_dict = VmVariantList(2)
      sub_dict.push_int(100)
      sub_dict.push_int(200)
      sub_list.push_list(sub_dict)
      sub_list.push_int(6)
      ret_list.push_list(sub_list)

    vm_context = MockVmContext(invoke)
    vm_function = MockVmFunction(
        reflection={
            "iree.abi":
                json.dumps({
                    "a": [],
                    "r": [
                        "i32",
                        [
                            "slist",
                            ["sdict", ["bar", "i32"], ["foo", "i32"]],
                            "i64",
                        ]
                    ],
                })
        })
    invoker = FunctionInvoker(vm_context, self.device, vm_function, tracer=None)
    result = invoker()
    self.assertEqual((3, [{'bar': 100, 'foo': 200}, 6]), result)

  def testMissingPositional(self):

    def invoke(arg_list, ret_list):
      ret_list.push_int(3)

    vm_context = MockVmContext(invoke)
    vm_function = MockVmFunction(
        reflection={
            "iree.abi":
                json.dumps({
                    "a": [
                        "i32",
                        ["named", "a", "i32"],
                        ["named", "b", "i32"],
                    ],
                    "r": ["i32",],
                })
        })
    invoker = FunctionInvoker(vm_context, self.device, vm_function, tracer=None)
    with self.assertRaisesRegexp(ValueError,
                                 "a required argument was not specified"):
      result = invoker(a=1, b=2)

  def testMissingKeyword(self):

    def invoke(arg_list, ret_list):
      ret_list.push_int(3)

    vm_context = MockVmContext(invoke)
    vm_function = MockVmFunction(
        reflection={
            "iree.abi":
                json.dumps({
                    "a": [
                        "i32",
                        ["named", "a", "i32"],
                        ["named", "b", "i32"],
                    ],
                    "r": ["i32",],
                })
        })
    invoker = FunctionInvoker(vm_context, self.device, vm_function, tracer=None)
    with self.assertRaisesRegexp(ValueError,
                                 "a required argument was not specified"):
      result = invoker(-1, a=1)

  def testExtraKeyword(self):

    def invoke(arg_list, ret_list):
      ret_list.push_int(3)

    vm_context = MockVmContext(invoke)
    vm_function = MockVmFunction(
        reflection={
            "iree.abi":
                json.dumps({
                    "a": [
                        "i32",
                        ["named", "a", "i32"],
                        ["named", "b", "i32"],
                    ],
                    "r": ["i32",],
                })
        })
    invoker = FunctionInvoker(vm_context, self.device, vm_function, tracer=None)
    with self.assertRaisesRegexp(ValueError, "specified kwarg 'c' is unknown"):
      result = invoker(-1, a=1, b=2, c=3)

  # TODO: Fill out all return types.
  def testReturnTypeNdArrayBool(self):
    result_array = np.asarray([1, 0], dtype=np.int8)

    def invoke(arg_list, ret_list):
      ret_list.push_buffer_view(self.device, result_array,
                                rt.HalElementType.UINT_8)

    vm_context = MockVmContext(invoke)
    vm_function = MockVmFunction(reflection={
        "iree.abi": json.dumps({
            "a": [],
            "r": [["ndarray", "i1", 1, 2]],
        })
    })
    invoker = FunctionInvoker(vm_context, self.device, vm_function, tracer=None)
    result = invoker()
    # assertEqual on bool arrays is fraught for... reasons.
    self.assertEqual("array([ True, False])", repr(result))

  def testReturnTypeList(self):
    vm_list = VmVariantList(2)
    vm_list.push_int(1)
    vm_list.push_int(2)

    def invoke(arg_list, ret_list):
      ret_list.push_list(vm_list)

    vm_context = MockVmContext(invoke)
    vm_function = MockVmFunction(reflection={
        "iree.abi":
            json.dumps({
                "a": [],
                "r": [["py_homogeneous_list", "i64"]],
            })
    })
    invoker = FunctionInvoker(vm_context, self.device, vm_function, tracer=None)
    result = invoker()
    self.assertEqual("[1, 2]", repr(result))


if __name__ == "__main__":
  absltest.main()
