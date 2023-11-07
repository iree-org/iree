# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import iree.compiler
import iree.runtime as rt


MM_TEST_COMPILED = None
MM_TEST_ASM = r"""
  #map = affine_map<(d0, d1) -> (d0, d1)>
  #map1 = affine_map<(d0, d1) -> (d1, d0)>
  #map2 = affine_map<(d0, d1) -> (d1)>
  module @external_global_parameters_map_dict {
    util.global private @_params.classifier.weight {noinline} = #stream.parameter.named<"params"::"weight"> : tensor<30x20xf32>
    util.global private @_params.classifier.bias {noinline} = #stream.parameter.named<"params"::"bias"> : tensor<30xf32>
  func.func @run(%arg0: tensor<128x20xf32>) -> tensor<128x30xf32> {
    %0 = call @forward(%arg0) : (tensor<128x20xf32>) -> tensor<128x30xf32>
    return %0 : tensor<128x30xf32>
  }
  func.func private @forward(%arg0: tensor<128x20xf32>) -> tensor<128x30xf32> attributes {torch.assume_strict_symbolic_shapes} {
    %cst = arith.constant 0.000000e+00 : f32
    %_params.classifier.weight = util.global.load @_params.classifier.weight : tensor<30x20xf32>
    %0 = tensor.empty() : tensor<20x30xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%_params.classifier.weight : tensor<30x20xf32>) outs(%0 : tensor<20x30xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<20x30xf32>
    %2 = tensor.empty() : tensor<128x30xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<128x30xf32>) -> tensor<128x30xf32>
    %4 = linalg.matmul ins(%arg0, %1 : tensor<128x20xf32>, tensor<20x30xf32>) outs(%3 : tensor<128x30xf32>) -> tensor<128x30xf32>
    %_params.classifier.bias = util.global.load @_params.classifier.bias : tensor<30xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %_params.classifier.bias : tensor<128x30xf32>, tensor<30xf32>) outs(%2 : tensor<128x30xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %6 = arith.addf %in, %in_0 : f32
      linalg.yield %6 : f32
    } -> tensor<128x30xf32>
    return %5 : tensor<128x30xf32>
  }
}
"""


def compile_mm_test():
    global MM_TEST_COMPILED
    if not MM_TEST_COMPILED:
        MM_TEST_COMPILED = iree.compiler.compile_str(
            MM_TEST_ASM, target_backends=iree.compiler.core.DEFAULT_TESTING_BACKENDS
        )
    return MM_TEST_COMPILED


def create_mm_test_module(instance):
    binary = compile_mm_test()
    return rt.VmModule.copy_buffer(instance, binary)


class ParameterTest(unittest.TestCase):
    def setUp(self):
        self.instance = rt.VmInstance()
        self.device = rt.get_device(iree.compiler.core.DEFAULT_TESTING_DRIVER)

    def testParameterIndex(self):
        index = rt.ParameterIndex()
        self.assertEqual(len(index), 0)
        index.reserve(25)
        self.assertEqual(len(index), 0)
        provider = index.create_provider()
        rt.create_io_parameters_module(self.instance, provider)
    def testBasic(self):
        empty_provider = rt.ParameterIndex().create_provider(scope="params")
        context = rt.VmContext(
            self.instance,
            [
                rt.create_io_parameters_module(self.instance, empty_provider),
                rt.create_hal_module(self.instance, self.device),
                create_mm_test_module(self.instance),
            ],
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
