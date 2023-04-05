// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

func.func @f1(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> (tensor<1xf32>, tensor<1xf32>) {
  %0 = arith.addf %arg0, %arg1: tensor<1xf32>
  return %0, %0 : tensor<1xf32>, tensor<1xf32>
}
