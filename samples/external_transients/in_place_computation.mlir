// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test for external transients feature: verifies that transient allocations
// can be emplaced into externally-provided storage.
//
// The test performs three sequential operations wrapped in dispatch regions
// to prevent fusion:
// 1. temp1 = input + 1.0  (uses input + transient storage)
// 2. temp2 = temp1 * 2.0  (uses transient + transient storage)
// 3. output = temp2 + input (uses transient + output buffer)
//
// With input=[1.0, 1.0, ...], the expected output is [5.0, 5.0, ...].

util.func public @in_place_computation(
  %input: tensor<64xf32>,
  %output: tensor<64xf32> {iree.abi.output = 0 : index},
  %transient_storage: !hal.buffer {iree.abi.transients}
) -> tensor<64xf32> {
  // Dispatch 1: temp1 = input + 1.0
  %temp1 = flow.dispatch.region -> (tensor<64xf32>) {
    %one = arith.constant dense<1.0> : tensor<64xf32>
    %result = arith.addf %input, %one : tensor<64xf32>
    flow.return %result : tensor<64xf32>
  }

  // Dispatch 2: temp2 = temp1 * 2.0
  %temp2 = flow.dispatch.region -> (tensor<64xf32>) {
    %two = arith.constant dense<2.0> : tensor<64xf32>
    %result = arith.mulf %temp1, %two : tensor<64xf32>
    flow.return %result : tensor<64xf32>
  }

  // Dispatch 3: output = temp2 + input
  %result = flow.dispatch.region -> (tensor<64xf32>) {
    %final = arith.addf %temp2, %input : tensor<64xf32>
    flow.return %final : tensor<64xf32>
  }

  util.return %result : tensor<64xf32>
}
