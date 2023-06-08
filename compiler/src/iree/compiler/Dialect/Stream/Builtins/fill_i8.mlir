// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// stream.builtin.fill.i8
// Writes the i8 %value %count times at byte %offset of %out_binding.

stream.executable private @__builtin_fill_i8 {
  stream.executable.export public @__builtin_fill_i8 workgroups(%arg0: index) -> (index, index, index) {
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg0
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @__builtin_fill_i8(%value: i8, %offset: index, %count: index, %out_binding: !stream.binding) {
      %out = stream.binding.subspan %out_binding[%offset] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<?xi8>>{%count}
      %0 = tensor.empty(%count) : tensor<?xi8>
      %1 = linalg.fill ins(%value : i8) outs(%0 : tensor<?xi8>) -> tensor<?xi8>
      flow.dispatch.tensor.store %1, %out, offsets = [0], sizes = [%count], strides = [1] : tensor<?xi8> -> !flow.dispatch.tensor<writeonly:tensor<?xi8>>{%count}
      return
    }
  }
}
