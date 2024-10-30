// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// stream.builtin.fill.i16
// Writes the i16 %value %count times in the bound range of %out_binding.

stream.executable private @__builtin_fill_i16 {
  stream.executable.export public @__builtin_fill_i16 workgroups(%arg0: index) -> (index, index, index) {
    %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @__builtin_fill_i16(%value: i16, %count: index, %out_binding: !stream.binding) {
      %c0 = arith.constant 0 : index
      %count0 = flow.dispatch.workload.ordinal %count, 0 : index
      %out = stream.binding.subspan %out_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<?xi16>>{%count0}
      %0 = tensor.empty(%count0) : tensor<?xi16>
      %1 = linalg.fill ins(%value : i16) outs(%0 : tensor<?xi16>) -> tensor<?xi16>
      flow.dispatch.tensor.store %1, %out, offsets = [0], sizes = [%count0], strides = [1] : tensor<?xi16> -> !flow.dispatch.tensor<writeonly:tensor<?xi16>>{%count}
      return
    }
  }
}
