// RUN: iree-opt --pass-pipeline="builtin.module(iree-dispatch-creation-hoist-encoding-ops,cse)" --split-input-file %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(iree-dispatch-creation-hoist-encoding-ops,cse)" --test-iree-dispatch-creation-no-hoist-data-operands-scaled-mma --split-input-file %s | FileCheck %s --check-prefix=PDT
// With --test-iree-dispatch-creation-no-hoist-data-operands-scaled-mma, only scale operands (indices 2, 3) of
// scaled_matmul are hoisted out of the dispatch. Data operands (0, 1) stay
// inside.

#sm0 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#sm1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#sm2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#sm3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#sm4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#enc0 = #iree_encoding.encoding<operand_index = 0 : index, op_type = scaled_matmul,
  element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32],
  user_indexing_maps = [#sm0, #sm1, #sm2, #sm3, #sm4]>
#enc1 = #iree_encoding.encoding<operand_index = 1 : index, op_type = scaled_matmul,
  element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32],
  user_indexing_maps = [#sm0, #sm1, #sm2, #sm3, #sm4]>
#enc2 = #iree_encoding.encoding<operand_index = 2 : index, op_type = scaled_matmul,
  element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32],
  user_indexing_maps = [#sm0, #sm1, #sm2, #sm3, #sm4]>
#enc3 = #iree_encoding.encoding<operand_index = 3 : index, op_type = scaled_matmul,
  element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32],
  user_indexing_maps = [#sm0, #sm1, #sm2, #sm3, #sm4]>
#enc4 = #iree_encoding.encoding<operand_index = 4 : index, op_type = scaled_matmul,
  element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32],
  user_indexing_maps = [#sm0, #sm1, #sm2, #sm3, #sm4]>

#gpu_target_950 = #iree_gpu.target<arch = "gfx950", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>
#exec_950 = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_950}>
#device_950 = #hal.device.target<"local", {ordinal = 0 : index}, [#exec_950]> : !hal.device

module {
  util.global private @device = #device_950
  util.func public @gfx950_hoist_scales_only(
      %lhs: tensor<256x128x32xf4E2M1FN>, %rhs: tensor<512x128x32xf4E2M1FN>,
      %lscale: tensor<256x128xf8E8M0FNU>, %rscale: tensor<512x128xf8E8M0FNU>)
      -> tensor<256x512xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %result = flow.dispatch.region -> (tensor<256x512xf32>) {
      %0 = iree_encoding.set_encoding %lhs : tensor<256x128x32xf4E2M1FN> -> tensor<256x128x32xf4E2M1FN, #enc0>
      %1 = iree_encoding.set_encoding %rhs : tensor<512x128x32xf4E2M1FN> -> tensor<512x128x32xf4E2M1FN, #enc1>
      %2 = iree_encoding.set_encoding %lscale : tensor<256x128xf8E8M0FNU> -> tensor<256x128xf8E8M0FNU, #enc2>
      %3 = iree_encoding.set_encoding %rscale : tensor<512x128xf8E8M0FNU> -> tensor<512x128xf8E8M0FNU, #enc3>
      %empty = tensor.empty() : tensor<256x512xf32, #enc4>
      %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<256x512xf32, #enc4>) -> tensor<256x512xf32, #enc4>
      %mm = linalg.generic {
          indexing_maps = [#sm0, #sm1, #sm2, #sm3, #sm4],
          iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
          ins(%0, %1, %2, %3 : tensor<256x128x32xf4E2M1FN, #enc0>, tensor<512x128x32xf4E2M1FN, #enc1>,
              tensor<256x128xf8E8M0FNU, #enc2>, tensor<512x128xf8E8M0FNU, #enc3>)
          outs(%fill : tensor<256x512xf32, #enc4>) {
      ^bb0(%a: f4E2M1FN, %b: f4E2M1FN, %c: f8E8M0FNU, %d: f8E8M0FNU, %e: f32):
        linalg.yield %e : f32
      } -> tensor<256x512xf32, #enc4>
      %out = iree_encoding.unset_encoding %mm : tensor<256x512xf32, #enc4> -> tensor<256x512xf32>
      flow.return %out : tensor<256x512xf32>
    }
    util.return %result : tensor<256x512xf32>
  }
}

// Without the flag, all set_encodings are hoisted.
// CHECK-LABEL: @gfx950_hoist_scales_only
// CHECK-SAME:    (%[[LHS:.+]]: tensor<{{.+}}f4E2M1FN>, %[[RHS:.+]]: tensor<{{.+}}f4E2M1FN>,
// CHECK-SAME:     %[[LS:.+]]: tensor<{{.+}}f8E8M0FNU>, %[[RS:.+]]: tensor<{{.+}}f8E8M0FNU>)
// CHECK-DAG:   iree_encoding.set_encoding %[[LHS]]
// CHECK-DAG:   iree_encoding.set_encoding %[[RHS]]
// CHECK-DAG:   iree_encoding.set_encoding %[[LS]]
// CHECK-DAG:   iree_encoding.set_encoding %[[RS]]
// CHECK:       flow.dispatch.region
// CHECK-NOT:     iree_encoding.set_encoding
// CHECK:         linalg.generic
// CHECK:       util.return

// With the flag: scale set_encodings (operands 2, 3)
// hoisted; data set_encodings (0, 1) stay inside the dispatch.
// PDT-LABEL: @gfx950_hoist_scales_only
// PDT-SAME:    (%[[LHS:.+]]: tensor<{{.+}}f4E2M1FN>, %[[RHS:.+]]: tensor<{{.+}}f4E2M1FN>,
// PDT-SAME:     %[[LS:.+]]: tensor<{{.+}}f8E8M0FNU>, %[[RS:.+]]: tensor<{{.+}}f8E8M0FNU>)
// PDT-DAG:   %[[ENC_LS:.+]] = iree_encoding.set_encoding %[[LS]]
// PDT-DAG:   %[[ENC_RS:.+]] = iree_encoding.set_encoding %[[RS]]
// PDT:       flow.dispatch.region
// PDT:         iree_encoding.set_encoding %[[LHS]]
// PDT:         iree_encoding.set_encoding %[[RHS]]
// PDT:         linalg.generic
// PDT:       util.return
