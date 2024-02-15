
// Preprocessing with generalized packing.
//
// RUN: iree-opt %s --iree-transform-dialect-interpreter --transform-dialect-drop-schedule | \
// RUN: iree-compile - --iree-hal-target-backends=llvm-cpu \
// RUN:     --iree-opt-data-tiling=false \
// RUN:     --compile-to=executable-configurations | \
// RUN: FileCheck %s

!a_tensor_t = tensor<1234x567xf32>
!b_tensor_t = tensor<567x890xf32>
!c_tensor_t = tensor<1234x890xf32>

// Note: the normalization in these maps is gone due to InterchangeGenericOps.
// When using generalized packing, it would be better to drop that pass.

// CHECK-LABEL: func.func @matmul_dispatch_0
//       CHECK:   tensor.empty() : tensor<155x18x8x32xf32>
//       CHECK:   tensor.pack

// CHECK-LABEL: func.func @matmul_dispatch_1
//       CHECK:   tensor.empty() : tensor<18x56x16x32xf32>
//       CHECK:   tensor.pack

// CHECK-LABEL: func.func @matmul_dispatch_2
//       CHECK:   tensor.empty() : tensor<155x56x8x16xf32>
//       CHECK:   tensor.pack

// CHECK-LABEL: func.func @matmul_dispatch_3
func.func public @matmul(%arg0: !a_tensor_t, %arg2: !c_tensor_t) -> !c_tensor_t {
  %rhs = arith.constant dense<0.1> : !b_tensor_t
  %c0 = util.optimization_barrier %rhs : !b_tensor_t
  //  CHECK-NOT: pack
  //      CHECK: linalg.generic
  // CHECK-SAME:   indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d2, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d1, d3, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>]
  // CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]}
  // CHECK-SAME:   ins(%{{.*}} : tensor<155x18x8x32xf32>, tensor<18x56x16x32xf32>)
  // CHECK-SAME:  outs(%{{.*}} : tensor<155x56x8x16xf32>)

  %0 = linalg.matmul
     ins(%arg0, %c0: !a_tensor_t, !b_tensor_t)
    outs(%arg2: !c_tensor_t) -> !c_tensor_t
  return %0 : !c_tensor_t
}

// CHECK-LABEL: func.func @matmul_dispatch_4
//       CHECK:   tensor.unpack
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match interface{LinalgOp} in %module_op
      : (!transform.any_op) -> (!transform.any_op)

    transform.structured.pack_greedily %matmul
        matmul_packed_sizes = [8, 16, 32]
        matmul_inner_dims_order = [0, 1, 2]
      : (!transform.any_op) -> !transform.op<"linalg.generic">
    transform.yield
  }
} // module
