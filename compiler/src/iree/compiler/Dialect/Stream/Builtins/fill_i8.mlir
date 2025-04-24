// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// stream.builtin.fill.i8
// Writes the i8 %value %count times in the bound range of %out_binding.

stream.executable private @__builtin_fill_i8 {
  stream.executable.export public @__builtin_fill_i8 workgroups(%arg0: index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice %arg0
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @__builtin_fill_i8(%value: i8, %count: index, %out_binding: !stream.binding) {
      %c0 = arith.constant 0 : index
      %count0 = iree_tensor_ext.dispatch.workload.ordinal %count, 0 : index
      %out = stream.binding.subspan %out_binding[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi8>>{%count0}
      %0 = tensor.empty(%count0) : tensor<?xi8>
      %1 = linalg.fill ins(%value : i8) outs(%0 : tensor<?xi8>) -> tensor<?xi8>
      iree_tensor_ext.dispatch.tensor.store %1, %out, offsets = [0], sizes = [%count0], strides = [1] : tensor<?xi8> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi8>>{%count}
      return
    }
  }
}
