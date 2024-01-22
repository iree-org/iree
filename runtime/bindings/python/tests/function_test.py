# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import numpy as np
import unittest

from iree import runtime as rt
from iree.runtime.function import (
    FunctionInvoker,
    IMPLICIT_BUFFER_ARG_MEMORY_TYPE,
    IMPLICIT_BUFFER_ARG_USAGE,
)
from iree.runtime._binding import VmVariantList


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


class FunctionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Doesn't matter what device. We just need one.
        config = rt.Config("local-task")
        cls.device = config.device

    def testNoReflectionScalars(self):
        def invoke(arg_list, ret_list):
            ret_list.push_int(3)
            ret_list.push_int(4)

        vm_context = MockVmContext(invoke)
        vm_function = MockVmFunction(reflection={})
        invoker = FunctionInvoker(vm_context, self.device, vm_function)
        result = invoker(1, 2)
        self.assertEqual("[<VmVariantList(2): [1, 2]>]", vm_context.mock_arg_reprs)
        self.assertEqual((3, 4), result)

    def testKeywordArgs(self):
        def invoke(arg_list, ret_list):
            ret_list.push_int(3)

        vm_context = MockVmContext(invoke)
        vm_function = MockVmFunction(
            reflection={
                "iree.abi": json.dumps(
                    {
                        "a": [
                            "i32",
                            ["named", "a", "i32"],
                            ["named", "b", "i32"],
                        ],
                        "r": [
                            "i32",
                        ],
                    }
                )
            }
        )
        invoker = FunctionInvoker(vm_context, self.device, vm_function)
        result = invoker(-1, a=1, b=2)
        self.assertEqual("[<VmVariantList(3): [-1, 1, 2]>]", vm_context.mock_arg_reprs)
        self.assertEqual(3, result)

    def testListArg(self):
        def invoke(arg_list, ret_list):
            ret_list.push_int(3)

        vm_context = MockVmContext(invoke)
        vm_function = MockVmFunction(
            reflection={
                "iree.abi": json.dumps(
                    {
                        "a": [
                            ["slist", "i32", "i32"],
                        ],
                        "r": [
                            "i32",
                        ],
                    }
                )
            }
        )
        invoker = FunctionInvoker(vm_context, self.device, vm_function)
        _ = invoker([2, 3])
        self.assertEqual(
            "[<VmVariantList(1): [List[2, 3]]>]", vm_context.mock_arg_reprs
        )

    def testListArgNoReflection(self):
        def invoke(arg_list, ret_list):
            ret_list.push_int(3)

        vm_context = MockVmContext(invoke)
        vm_function = MockVmFunction(reflection={})
        invoker = FunctionInvoker(vm_context, self.device, vm_function)
        _ = invoker([2, 3])
        self.assertEqual(
            "[<VmVariantList(1): [List[2, 3]]>]", vm_context.mock_arg_reprs
        )

    def testListArgArityMismatch(self):
        def invoke(arg_list, ret_list):
            ret_list.push_int(3)

        vm_context = MockVmContext(invoke)
        vm_function = MockVmFunction(
            reflection={
                "iree.abi": json.dumps(
                    {
                        "a": [
                            ["slist", "i32", "i32"],
                        ],
                        "r": [
                            "i32",
                        ],
                    }
                )
            }
        )
        invoker = FunctionInvoker(vm_context, self.device, vm_function)
        with self.assertRaisesRegex(
            ValueError, "expected a sequence with 2 values. got:"
        ):
            _ = invoker([2, 3, 4])

    def testTupleArg(self):
        def invoke(arg_list, ret_list):
            ret_list.push_int(3)

        vm_context = MockVmContext(invoke)
        vm_function = MockVmFunction(
            reflection={
                "iree.abi": json.dumps(
                    {
                        "a": [
                            ["stuple", "i32", "i32"],
                        ],
                        "r": [
                            "i32",
                        ],
                    }
                )
            }
        )
        invoker = FunctionInvoker(vm_context, self.device, vm_function)
        _ = invoker((2, 3))
        self.assertEqual(
            "[<VmVariantList(1): [List[2, 3]]>]", vm_context.mock_arg_reprs
        )

    def testDictArg(self):
        def invoke(arg_list, ret_list):
            ret_list.push_int(3)

        vm_context = MockVmContext(invoke)
        vm_function = MockVmFunction(
            reflection={
                "iree.abi": json.dumps(
                    {
                        "a": [
                            ["sdict", ["a", "i32"], ["b", "i32"]],
                        ],
                        "r": [
                            "i32",
                        ],
                    }
                )
            }
        )
        invoker = FunctionInvoker(vm_context, self.device, vm_function)
        _ = invoker({"b": 3, "a": 2})
        self.assertEqual(
            "[<VmVariantList(1): [List[2, 3]]>]", vm_context.mock_arg_reprs
        )

    def testDictArgArityMismatch(self):
        def invoke(arg_list, ret_list):
            ret_list.push_int(3)

        vm_context = MockVmContext(invoke)
        vm_function = MockVmFunction(
            reflection={
                "iree.abi": json.dumps(
                    {
                        "a": [
                            ["sdict", ["a", "i32"], ["b", "i32"]],
                        ],
                        "r": [
                            "i32",
                        ],
                    }
                )
            }
        )
        invoker = FunctionInvoker(vm_context, self.device, vm_function)
        with self.assertRaisesRegex(ValueError, "expected a dict with 2 values. got:"):
            _ = invoker({"a": 2, "b": 3, "c": 4})

    def testDictArgKeyError(self):
        def invoke(arg_list, ret_list):
            ret_list.push_int(3)

        vm_context = MockVmContext(invoke)
        vm_function = MockVmFunction(
            reflection={
                "iree.abi": json.dumps(
                    {
                        "a": [
                            ["sdict", ["a", "i32"], ["b", "i32"]],
                        ],
                        "r": [
                            "i32",
                        ],
                    }
                )
            }
        )
        invoker = FunctionInvoker(vm_context, self.device, vm_function)
        with self.assertRaisesRegex(ValueError, "could not get item 'b' from: "):
            _ = invoker({"a": 2, "c": 3})

    def testDictArgNoReflection(self):
        def invoke(arg_list, ret_list):
            ret_list.push_int(3)

        vm_context = MockVmContext(invoke)
        vm_function = MockVmFunction(reflection={})
        invoker = FunctionInvoker(vm_context, self.device, vm_function)
        _ = invoker({"b": 3, "a": 2})
        self.assertEqual(
            "[<VmVariantList(1): [List[2, 3]]>]", vm_context.mock_arg_reprs
        )

    def testInlinedResults(self):
        def invoke(arg_list, ret_list):
            ret_list.push_int(3)
            ret_list.push_int(4)

        vm_context = MockVmContext(invoke)
        vm_function = MockVmFunction(
            reflection={
                "iree.abi": json.dumps(
                    {
                        "a": [],
                        "r": [["slist", "i32", "i32"]],
                    }
                )
            }
        )
        invoker = FunctionInvoker(vm_context, self.device, vm_function)
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
                "iree.abi": json.dumps(
                    {
                        "a": [],
                        "r": [
                            "i32",
                            [
                                "slist",
                                ["sdict", ["bar", "i32"], ["foo", "i32"]],
                                "i64",
                            ],
                        ],
                    }
                )
            }
        )
        invoker = FunctionInvoker(vm_context, self.device, vm_function)
        result = invoker()
        self.assertEqual((3, [{"bar": 100, "foo": 200}, 6]), result)

    def testMissingPositional(self):
        def invoke(arg_list, ret_list):
            ret_list.push_int(3)

        vm_context = MockVmContext(invoke)
        vm_function = MockVmFunction(
            reflection={
                "iree.abi": json.dumps(
                    {
                        "a": [
                            "i32",
                            ["named", "a", "i32"],
                            ["named", "b", "i32"],
                        ],
                        "r": [
                            "i32",
                        ],
                    }
                )
            }
        )
        invoker = FunctionInvoker(vm_context, self.device, vm_function)
        with self.assertRaisesRegex(ValueError, "mismatched call arity:"):
            result = invoker(a=1, b=1)

    def testMissingPositionalNdarray(self):
        def invoke(arg_list, ret_list):
            ret_list.push_int(3)

        vm_context = MockVmContext(invoke)
        vm_function = MockVmFunction(
            reflection={
                "iree.abi": json.dumps(
                    {
                        "a": [
                            ["ndarray", "i32", 1, 1],
                            ["named", "a", ["ndarray", "i32", 1, 1]],
                            ["named", "b", ["ndarray", "i32", 1, 1]],
                        ],
                        "r": [
                            "i32",
                        ],
                    }
                )
            }
        )
        invoker = FunctionInvoker(vm_context, self.device, vm_function)
        with self.assertRaisesRegex(ValueError, "mismatched call arity:"):
            result = invoker(a=1, b=1)

    def testMissingKeyword(self):
        def invoke(arg_list, ret_list):
            ret_list.push_int(3)

        vm_context = MockVmContext(invoke)
        vm_function = MockVmFunction(
            reflection={
                "iree.abi": json.dumps(
                    {
                        "a": [
                            "i32",
                            ["named", "a", "i32"],
                            ["named", "b", "i32"],
                        ],
                        "r": [
                            "i32",
                        ],
                    }
                )
            }
        )
        invoker = FunctionInvoker(vm_context, self.device, vm_function)
        with self.assertRaisesRegex(ValueError, "mismatched call arity:"):
            result = invoker(-1, a=1)

    def testMissingKeywordNdArray(self):
        def invoke(arg_list, ret_list):
            ret_list.push_int(3)

        vm_context = MockVmContext(invoke)
        vm_function = MockVmFunction(
            reflection={
                "iree.abi": json.dumps(
                    {
                        "a": [
                            ["ndarray", "i32", 1, 1],
                            ["named", "a", ["ndarray", "i32", 1, 1]],
                            ["named", "b", ["ndarray", "i32", 1, 1]],
                        ],
                        "r": [
                            "i32",
                        ],
                    }
                )
            }
        )
        invoker = FunctionInvoker(vm_context, self.device, vm_function)
        with self.assertRaisesRegex(ValueError, "mismatched call arity:"):
            result = invoker(-1, a=1)

    def testExtraKeyword(self):
        def invoke(arg_list, ret_list):
            ret_list.push_int(3)

        vm_context = MockVmContext(invoke)
        vm_function = MockVmFunction(
            reflection={
                "iree.abi": json.dumps(
                    {
                        "a": [
                            "i32",
                            ["named", "a", "i32"],
                            ["named", "b", "i32"],
                        ],
                        "r": [
                            "i32",
                        ],
                    }
                )
            }
        )
        invoker = FunctionInvoker(vm_context, self.device, vm_function)
        with self.assertRaisesRegex(ValueError, "specified kwarg 'c' is unknown"):
            result = invoker(-1, a=1, b=2, c=3)

    def testNdarrayArg(self):
        arg_array = np.asarray([1, 0], dtype=np.int32)

        invoked_arg_list = None

        def invoke(arg_list, ret_list):
            nonlocal invoked_arg_list
            invoked_arg_list = arg_list

        vm_context = MockVmContext(invoke)
        vm_function = MockVmFunction(
            reflection={
                "iree.abi": json.dumps(
                    {
                        "a": [["ndarray", "i32", 1, 2]],
                        "r": [],
                    }
                )
            }
        )
        invoker = FunctionInvoker(vm_context, self.device, vm_function)
        result = invoker(arg_array)
        self.assertEqual(
            "<VmVariantList(1): [HalBufferView(2:0x20000011)]>", repr(invoked_arg_list)
        )

    def testDeviceArrayArg(self):
        # Note that since the device array is set up to disallow implicit host
        # transfers, this also verifies that no accidental/automatic transfers
        # are done as part of marshalling the array to the function.
        arg_array = rt.asdevicearray(
            self.device,
            np.asarray([1, 0], dtype=np.int32),
            implicit_host_transfer=False,
        )

        invoked_arg_list = None

        def invoke(arg_list, ret_list):
            nonlocal invoked_arg_list
            invoked_arg_list = arg_list

        vm_context = MockVmContext(invoke)
        vm_function = MockVmFunction(
            reflection={
                "iree.abi": json.dumps(
                    {
                        "a": [["ndarray", "i32", 1, 2]],
                        "r": [],
                    }
                )
            }
        )
        invoker = FunctionInvoker(vm_context, self.device, vm_function)
        result = invoker(arg_array)
        self.assertEqual(
            "<VmVariantList(1): [HalBufferView(2:0x20000011)]>", repr(invoked_arg_list)
        )

    def testBufferViewArg(self):
        arg_buffer_view = self.device.allocator.allocate_buffer_copy(
            memory_type=IMPLICIT_BUFFER_ARG_MEMORY_TYPE,
            allowed_usage=IMPLICIT_BUFFER_ARG_USAGE,
            device=self.device,
            buffer=np.asarray([1, 0], dtype=np.int32),
            element_type=rt.HalElementType.SINT_32,
        )

        invoked_arg_list = None

        def invoke(arg_list, ret_list):
            nonlocal invoked_arg_list
            invoked_arg_list = arg_list

        vm_context = MockVmContext(invoke)
        vm_function = MockVmFunction(
            reflection={
                "iree.abi": json.dumps(
                    {
                        "a": [["ndarray", "i32", 1, 2]],
                        "r": [],
                    }
                )
            }
        )
        invoker = FunctionInvoker(vm_context, self.device, vm_function)
        _ = invoker(arg_buffer_view)
        self.assertEqual(
            "<VmVariantList(1): [HalBufferView(2:0x20000011)]>", repr(invoked_arg_list)
        )

    def testNdarrayArgNoReflection(self):
        arg_array = np.asarray([1, 0], dtype=np.int32)

        invoked_arg_list = None

        def invoke(arg_list, ret_list):
            nonlocal invoked_arg_list
            invoked_arg_list = arg_list

        vm_context = MockVmContext(invoke)
        vm_function = MockVmFunction(reflection={})
        invoker = FunctionInvoker(vm_context, self.device, vm_function)
        result = invoker(arg_array)
        self.assertEqual(
            "<VmVariantList(1): [HalBufferView(2:0x20000011)]>", repr(invoked_arg_list)
        )

    def testDeviceArrayArgNoReflection(self):
        # Note that since the device array is set up to disallow implicit host
        # transfers, this also verifies that no accidental/automatic transfers
        # are done as part of marshalling the array to the function.
        arg_array = rt.asdevicearray(
            self.device,
            np.asarray([1, 0], dtype=np.int32),
            implicit_host_transfer=False,
        )

        invoked_arg_list = None

        def invoke(arg_list, ret_list):
            nonlocal invoked_arg_list
            invoked_arg_list = arg_list

        vm_context = MockVmContext(invoke)
        vm_function = MockVmFunction(reflection={})
        invoker = FunctionInvoker(vm_context, self.device, vm_function)
        result = invoker(arg_array)
        self.assertEqual(
            "<VmVariantList(1): [HalBufferView(2:0x20000011)]>", repr(invoked_arg_list)
        )

    def testBufferViewArgNoReflection(self):
        arg_buffer_view = self.device.allocator.allocate_buffer_copy(
            memory_type=IMPLICIT_BUFFER_ARG_MEMORY_TYPE,
            allowed_usage=IMPLICIT_BUFFER_ARG_USAGE,
            device=self.device,
            buffer=np.asarray([1, 0], dtype=np.int32),
            element_type=rt.HalElementType.SINT_32,
        )

        invoked_arg_list = None

        def invoke(arg_list, ret_list):
            nonlocal invoked_arg_list
            invoked_arg_list = arg_list

        vm_context = MockVmContext(invoke)
        vm_function = MockVmFunction(reflection={})
        invoker = FunctionInvoker(vm_context, self.device, vm_function)
        _ = invoker(arg_buffer_view)
        self.assertEqual(
            "<VmVariantList(1): [HalBufferView(2:0x20000011)]>", repr(invoked_arg_list)
        )

    def testReturnBufferView(self):
        result_array = np.asarray([1, 0], dtype=np.int32)

        def invoke(arg_list, ret_list):
            buffer_view = self.device.allocator.allocate_buffer_copy(
                memory_type=IMPLICIT_BUFFER_ARG_MEMORY_TYPE,
                allowed_usage=IMPLICIT_BUFFER_ARG_USAGE,
                device=self.device,
                buffer=result_array,
                element_type=rt.HalElementType.SINT_32,
            )
            ret_list.push_ref(buffer_view)

        vm_context = MockVmContext(invoke)
        vm_function = MockVmFunction(
            reflection={
                "iree.abi": json.dumps(
                    {
                        "a": [],
                        "r": [["ndarray", "i32", 1, 2]],
                    }
                )
            }
        )
        invoker = FunctionInvoker(vm_context, self.device, vm_function)
        result = invoker()
        np.testing.assert_array_equal([1, 0], result)

    def testReturnBufferViewNoReflection(self):
        result_array = np.asarray([1, 0], dtype=np.int32)

        def invoke(arg_list, ret_list):
            buffer_view = self.device.allocator.allocate_buffer_copy(
                memory_type=IMPLICIT_BUFFER_ARG_MEMORY_TYPE,
                allowed_usage=IMPLICIT_BUFFER_ARG_USAGE,
                device=self.device,
                buffer=result_array,
                element_type=rt.HalElementType.SINT_32,
            )
            ret_list.push_ref(buffer_view)

        vm_context = MockVmContext(invoke)
        vm_function = MockVmFunction(reflection={})
        invoker = FunctionInvoker(vm_context, self.device, vm_function)
        result = invoker()
        np.testing.assert_array_equal([1, 0], result)

    # TODO: Fill out all return types.
    def testReturnTypeNdArrayBool(self):
        result_array = np.asarray([1, 0], dtype=np.int8)

        def invoke(arg_list, ret_list):
            buffer_view = self.device.allocator.allocate_buffer_copy(
                memory_type=IMPLICIT_BUFFER_ARG_MEMORY_TYPE,
                allowed_usage=IMPLICIT_BUFFER_ARG_USAGE,
                device=self.device,
                buffer=result_array,
                element_type=rt.HalElementType.UINT_8,
            )
            ret_list.push_ref(buffer_view)

        vm_context = MockVmContext(invoke)
        vm_function = MockVmFunction(
            reflection={
                "iree.abi": json.dumps(
                    {
                        "a": [],
                        "r": [["ndarray", "i1", 1, 2]],
                    }
                )
            }
        )
        invoker = FunctionInvoker(vm_context, self.device, vm_function)
        result = invoker()
        # assertEqual on bool arrays is fraught for... reasons.
        np.testing.assert_array_equal([True, False], result)

    def testReturnTypeList(self):
        vm_list = VmVariantList(2)
        vm_list.push_int(1)
        vm_list.push_int(2)

        def invoke(arg_list, ret_list):
            ret_list.push_list(vm_list)

        vm_context = MockVmContext(invoke)
        vm_function = MockVmFunction(
            reflection={
                "iree.abi": json.dumps(
                    {
                        "a": [],
                        "r": [["py_homogeneous_list", "i64"]],
                    }
                )
            }
        )
        invoker = FunctionInvoker(vm_context, self.device, vm_function)
        result = invoker()
        self.assertEqual("[1, 2]", repr(result))


if __name__ == "__main__":
    unittest.main()
