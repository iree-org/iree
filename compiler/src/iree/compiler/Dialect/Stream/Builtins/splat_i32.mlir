// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// stream.builtin.splat.i32
// Writes the i32 %value %count times at offset 0 of %out_binding.

stream.executable private @__builtin_splat_i32 {
  stream.executable.export public @__builtin_splat_i32 workgroups(%arg0: index) -> (index, index, index) {
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg0
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @__builtin_splat_i32(%value: i32, %count: index, %out_binding: !stream.binding) {
      %c0 = arith.constant 0 : index
      %out = stream.binding.subspan %out_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<?xi32>>{%count}
      %0 = tensor.empty(%count) : tensor<?xi32>
      %1 = linalg.fill ins(%value : i32) outs(%0 : tensor<?xi32>) -> tensor<?xi32>
      flow.dispatch.tensor.store %1, %out, offsets = [0], sizes = [%count], strides = [1] : tensor<?xi32> -> !flow.dispatch.tensor<writeonly:tensor<?xi32>>{%count}
      return
    }
  }
}
