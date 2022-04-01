// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// stream.builtin.fill.i64
// Writes the i64 %value %count times at byte %offset of %out_binding.

stream.executable private @__builtin_fill_i64 {
  stream.executable.export public @__builtin_fill_i64
  builtin.module {
    func.func @__builtin_fill_i64(%value: i64, %offset: index, %count: index, %out_binding: !stream.binding) {
      %out = stream.binding.subspan %out_binding[%offset] : !stream.binding -> !flow.dispatch.tensor<writeonly:?xi64>{%count}
      %workgroup_size_0 = flow.dispatch.workgroup.size[0] : index
      %workgroup_id_0 = flow.dispatch.workgroup.id[0] : index
      %workgroup_count_0 = flow.dispatch.workgroup.count[0] : index
      %1 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_0, %workgroup_size_0]
      %2 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_0, %workgroup_size_0]
      scf.for %i = %1 to %count step %2 {
        %3 = affine.min affine_map<(d0)[s0, s1] -> (s1, -d0 + s0)>(%i)[%count, %workgroup_size_0]
        %4 = linalg.init_tensor [%3] : tensor<?xi64>
        %5 = linalg.fill ins(%value : i64) outs(%4 : tensor<?xi64>) -> tensor<?xi64>
        flow.dispatch.tensor.store %5, %out, offsets = [%i], sizes = [%3], strides = [1] : tensor<?xi64> -> !flow.dispatch.tensor<writeonly:?xi64>{%count}
      }
      return
    }
  }
}
