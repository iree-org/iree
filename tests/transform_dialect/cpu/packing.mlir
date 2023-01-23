// RUN: iree-opt %s --iree-transform-dialect-interpreter --transform-dialect-drop-schedule --split-input-file | FileCheck %s

!A_mk = tensor<1023x255xf32>
!B_kn = tensor<255x127xf32>
!C_mn = tensor<1023x127xf32>

// CHECK-DAG: #[[$mk_kkmm:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d5, d3)>
// CHECK-DAG: #[[$kn_kknn:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// CHECK-DAG: #[[$mn_mmnn:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-LABEL: @matmul_mk_kn_mn(
func.func @matmul_mk_kn_mn(%A : !A_mk, %B : !B_kn, %C : !C_mn) -> !C_mn {
  //      CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#[[$mk_kkmm]], #[[$kn_kknn]], #[[$mn_mmnn]]],
  // CHECK-SAME:   ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} 
  // CHECK-SAME:   ins(%{{.*}} : tensor<128x8x32x8xf32>, tensor<8x8x32x16xf32>)
  // CHECK-SAME:  outs(%{{.*}} : tensor<128x8x8x16xf32>)
  %0 = linalg.matmul ins(%A, %B : !A_mk, !B_kn) outs(%C : !C_mn) -> !C_mn
  return %0 : !C_mn
}

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %func = transform.structured.match ops{["func.func"]} in %module_op
  %func_2 = transform.iree.pack_greedily %func
}

// -----

// !A_mk defined above
!A_mk = tensor<1023x255xf32>
!B_nk = tensor<127x255xf32>
!C_nm = tensor<127x1023xf32>

#mkn_accesses = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (n, m)>
]
#mkn_trait = {
  indexing_maps = #mkn_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-DAG: #[[$mk_kkmm:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d5, d3)>
// CHECK-DAG: #[[$kn_kknn:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// CHECK-DAG: #[[$mn_mmnn:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-LABEL: @matmul_mk_nk_nm(
func.func @matmul_mk_nk_nm(%A : !A_mk, %B : !B_nk, %C : !C_nm) -> !C_nm {
  //      CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#[[$mk_kkmm]], #[[$kn_kknn]], #[[$mn_mmnn]]],
  // CHECK-SAME:   ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} 
  // CHECK-SAME:   ins(%{{.*}} : tensor<128x8x32x8xf32>, tensor<8x8x32x16xf32>)
  // CHECK-SAME:  outs(%{{.*}} : tensor<128x8x8x16xf32>)
  %0 = linalg.generic #mkn_trait ins(%A, %B : !A_mk, !B_nk) outs(%C : !C_nm) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %d = arith.mulf %a, %b : f32
      %e = arith.addf %c, %d : f32
      linalg.yield %e : f32
  } -> !C_nm
  return %0 : !C_nm
}

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %func = transform.structured.match ops{["func.func"]} in %module_op
  %func_2 = transform.iree.pack_greedily %func
}

// -----

// !A_mk defined above
!A_mk = tensor<1023x255xf32>
!B_nk = tensor<127x255xf32>
!C_nm = tensor<127x1023xf32>

#mkn_accesses = [
  affine_map<(k, m, n) -> (m, k)>,
  affine_map<(k, m, n) -> (n, k)>,
  affine_map<(k, m, n) -> (n, m)>
]
#mkn_trait = {
  indexing_maps = #mkn_accesses,
  iterator_types = ["reduction", "parallel", "parallel"]
}

// CHECK-DAG: #[[$mk_kkmm:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d5, d3)>
// CHECK-DAG: #[[$kn_kknn:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// CHECK-DAG: #[[$mn_mmnn:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-LABEL: @matmul_mk_nk_nm_transposed(
func.func @matmul_mk_nk_nm_transposed(%A : !A_mk, %B : !B_nk, %C : !C_nm) -> !C_nm {
  //      CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#[[$mk_kkmm]], #[[$kn_kknn]], #[[$mn_mmnn]]],
  // CHECK-SAME:   ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} 
  // CHECK-SAME:   ins(%{{.*}} : tensor<128x8x32x8xf32>, tensor<8x8x32x16xf32>)
  // CHECK-SAME:  outs(%{{.*}} : tensor<128x8x8x16xf32>)
  %0 = linalg.generic #mkn_trait ins(%A, %B : !A_mk, !B_nk) outs(%C : !C_nm) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %d = arith.mulf %a, %b : f32
      %e = arith.addf %c, %d : f32
      linalg.yield %e : f32
  } -> !C_nm
  return %0 : !C_nm
}

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %func = transform.structured.match ops{["func.func"]} in %module_op
  %func_2 = transform.iree.pack_greedily %func
}
