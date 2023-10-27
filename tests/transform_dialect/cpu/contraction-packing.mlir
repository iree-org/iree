
// Preprocessing with generalized packing.
//
// RUN: iree-opt %s --iree-transform-dialect-interpreter --transform-dialect-drop-schedule | \
// RUN: FileCheck %s

!a_tensor_t = tensor<1234x567xf32>
!at_tensor_t = tensor<567x1234xf32>
!b_tensor_t = tensor<567x890xf32>
!bt_tensor_t = tensor<890x567xf32>
!c_tensor_t = tensor<1234x890xf32>
!ct_tensor_t = tensor<890x1234xf32>

// CHECK-DAG: #[[$map_lhs:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[$map_rhs:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d4, d5)>
// CHECK-DAG: #[[$map_res:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// CHECK-DAG: #[[$map_tlhs:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
// CHECK-DAG: #[[$map_trhs:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
// CHECK-DAG: #[[$map_tres:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>

// CHECK-LABEL: func.func @matmul_nnn
func.func @matmul_nnn(%arg0: !a_tensor_t, %arg2: !c_tensor_t) -> !c_tensor_t {
  %c0 = arith.constant dense<0.1> : !b_tensor_t

  //      CHECK: tensor.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [8, 32]
  //      CHECK: tensor.pack %{{.*}} inner_dims_pos = [1, 0] inner_tiles = [16, 32]
  //      CHECK: tensor.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [8, 16]
  //      CHECK: linalg.generic
  // CHECK-SAME:   indexing_maps = [#[[$map_lhs]], #[[$map_rhs]], #[[$map_res]]] 
  // CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
  // CHECK-SAME:   ins(%{{.*}} : tensor<155x18x8x32xf32>, tensor<18x56x16x32xf32>) 
  // CHECK-SAME:  outs(%{{.*}} : tensor<155x56x8x16xf32>)
  //      CHECK: tensor.unpack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [8, 16]
  %0 = linalg.matmul
     ins(%arg0, %c0: !a_tensor_t, !b_tensor_t)
    outs(%arg2: !c_tensor_t) -> !c_tensor_t
  return %0 : !c_tensor_t
}

#matmul_tnn_trait = {
  indexing_maps = [
    affine_map<(m, n, k) -> (k, m)>,
    affine_map<(m, n, k) -> (k, n)>,
    affine_map<(m, n, k) -> (m, n)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func.func @matmul_tnn
func.func @matmul_tnn(%arg0: !at_tensor_t, %arg2: !c_tensor_t) -> !c_tensor_t {
  %c0 = arith.constant dense<0.1> : !b_tensor_t

  //      CHECK: tensor.pack %{{.*}} inner_dims_pos = [1, 0] inner_tiles = [8, 32]
  //      CHECK: tensor.pack %{{.*}} inner_dims_pos = [1, 0] inner_tiles = [16, 32]
  //      CHECK: tensor.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [8, 16]
  //      CHECK: linalg.generic
  // CHECK-SAME:   indexing_maps = [#[[$map_tlhs]], #[[$map_rhs]], #[[$map_res]]] 
  // CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
  // CHECK-SAME:   ins(%{{.*}} : tensor<18x155x8x32xf32>, tensor<18x56x16x32xf32>) 
  // CHECK-SAME:  outs(%{{.*}} : tensor<155x56x8x16xf32>)
  //      CHECK: tensor.unpack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [8, 16]
  %0 = linalg.generic #matmul_tnn_trait
     ins(%arg0, %c0: !at_tensor_t, !b_tensor_t)
    outs(%arg2: !c_tensor_t) {
    ^bb(%a: f32, %b: f32, %c: f32) :
      %d = arith.mulf %a, %b: f32
      %e = arith.addf %c, %d: f32
      linalg.yield %e : f32
  } -> !c_tensor_t
  return %0 : !c_tensor_t
}

#matmul_ntn_trait = {
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (n, k)>,
    affine_map<(m, n, k) -> (m, n)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func.func @matmul_ntn
func.func @matmul_ntn(%arg0: !a_tensor_t, %arg2: !c_tensor_t) -> !c_tensor_t {
  %c0 = arith.constant dense<0.1> : !bt_tensor_t

  //      CHECK: tensor.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [8, 32]
  //      CHECK: tensor.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [16, 32]
  //      CHECK: tensor.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [8, 16]
  //      CHECK: linalg.generic
  // CHECK-SAME:   indexing_maps = [#[[$map_lhs]], #[[$map_trhs]], #[[$map_res]]] 
  // CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
  // CHECK-SAME:   ins(%{{.*}} : tensor<155x18x8x32xf32>, tensor<56x18x16x32xf32>) 
  // CHECK-SAME:  outs(%{{.*}} : tensor<155x56x8x16xf32>)
  //      CHECK: tensor.unpack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [8, 16]
  %0 = linalg.generic #matmul_ntn_trait
     ins(%arg0, %c0: !a_tensor_t, !bt_tensor_t)
    outs(%arg2: !c_tensor_t) {
    ^bb(%a: f32, %b: f32, %c: f32) :
      %d = arith.mulf %a, %b: f32
      %e = arith.addf %c, %d: f32
      linalg.yield %e : f32
  } -> !c_tensor_t
  return %0 : !c_tensor_t
}

#matmul_nnt_trait = {
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (k, n)>,
    affine_map<(m, n, k) -> (n, m)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func.func @matmul_nnt
func.func @matmul_nnt(%arg0: !a_tensor_t, %arg2: !ct_tensor_t) -> !ct_tensor_t {
  %c0 = arith.constant dense<0.1> : !b_tensor_t

  //      CHECK: tensor.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [8, 32]
  //      CHECK: tensor.pack %{{.*}} inner_dims_pos = [1, 0] inner_tiles = [16, 32]
  //      CHECK: tensor.pack %{{.*}} inner_dims_pos = [1, 0] inner_tiles = [8, 16]
  //      CHECK: linalg.generic
  // CHECK-SAME:   indexing_maps = [#[[$map_lhs]], #[[$map_rhs]], #[[$map_tres]]] 
  // CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
  // CHECK-SAME:   ins(%{{.*}} : tensor<155x18x8x32xf32>, tensor<18x56x16x32xf32>) 
  // CHECK-SAME:  outs(%{{.*}} : tensor<56x155x8x16xf32>)
  //      CHECK: tensor.unpack %{{.*}} inner_dims_pos = [1, 0] inner_tiles = [8, 16]
  %0 = linalg.generic #matmul_nnt_trait
     ins(%arg0, %c0: !a_tensor_t, !b_tensor_t)
    outs(%arg2: !ct_tensor_t) {
    ^bb(%a: f32, %b: f32, %c: f32) :
      %d = arith.mulf %a, %b: f32
      %e = arith.addf %c, %d: f32
      linalg.yield %e : f32
  } -> !ct_tensor_t
  return %0 : !ct_tensor_t
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match interface{LinalgOp} in %module_op
      : (!transform.any_op) -> (!transform.any_op)
    
    // Generalized packing rewrite extracts a gemm from any linalg op that contains 
    // one. This acts as a powerful normalization step: after this point, we have a
    // gemm (i.e. 3-D contraction with (m,n,k)=(8,16,32) ) on the 3 most minor
    // dimensions.
    transform.structured.pack_greedily %matmul
        matmul_packed_sizes = [8, 16, 32] matmul_inner_dims_order = [0, 1, 2]
      : (!transform.any_op) -> !transform.op<"linalg.generic">
    transform.yield 
  }
} // module

