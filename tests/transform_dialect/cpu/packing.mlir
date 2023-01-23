// // RUN: iree-opt %s --iree-transform-dialect-interpreter --transform-dialect-drop-schedule --split-input-file | FileCheck %s

!A_mk = tensor<1023x255xf32>
!B_kn = tensor<255x127xf32>
!C_mn = tensor<1023x127xf32>

// Normalized dims pre-packing are:         ( k,  m,  n)
// CHECK-DAG: #[[$mk_kkmm:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
// CHECK-DAG: #[[$kn_kknn:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[$mn_mmnn:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>

// CHECK-LABEL: @matmul_mk_kn_mn(
func.func @matmul_mk_kn_mn(%A : !A_mk, %B : !B_kn, %C : !C_mn) -> !C_mn {
  //      CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#[[$mk_kkmm]], #[[$kn_kknn]], #[[$mn_mmnn]]],
  // CHECK-SAME:   ["reduction", "parallel", "parallel", "reduction", "parallel", "parallel"]} 
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

// Normalized dims pre-packing are:         ( k,  m,  n)
// CHECK-DAG: #[[$km_kkmm:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
// CHECK-DAG: #[[$kn_kknn:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
// CHECK-DAG: #[[$mn_mmnn:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d4, d5)>

// CHECK-LABEL: @matmul_mk_nk_nm(
func.func @matmul_mk_nk_nm(%A : !A_mk, %B : !B_nk, %C : !C_nm) -> !C_nm {
  //      CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#[[$mk_kkmm]], #[[$kn_kknn]], #[[$mn_mmnn]]],
  // CHECK-SAME:   ["reduction", "parallel", "parallel", "reduction", "parallel", "parallel"]} 
  // CHECK-SAME:   ins(%{{.*}} : tensor<128x8x32x8xf32>, tensor<8x8x32x16xf32>)
  // CHECK-SAME:  outs(%{{.*}} : tensor<8x128x8x16xf32>)
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

// CHECK-DAG: #[[$mk_kkmm:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
// CHECK-DAG: #[[$kn_kknn:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
// CHECK-DAG: #[[$mn_mmnn:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d4, d5)>

// CHECK-LABEL: @matmul_mk_nk_nm_transposed(
func.func @matmul_mk_nk_nm_transposed(%A : !A_mk, %B : !B_nk, %C : !C_nm) -> !C_nm {
  //      CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#[[$mk_kkmm]], #[[$kn_kknn]], #[[$mn_mmnn]]],
  // CHECK-SAME:   ["reduction", "parallel", "parallel", "reduction", "parallel", "parallel"]} 
  // CHECK-SAME:   ins(%{{.*}} : tensor<128x8x32x8xf32>, tensor<8x8x32x16xf32>)
  // CHECK-SAME:  outs(%{{.*}} : tensor<8x128x8x16xf32>)
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

!A_bmkm2 = tensor<42x1023x255x33xf32>
!B_nkb = tensor<127x255x42xf32>
!C_nbm = tensor<127x42x1023xf32>

#mkn_accesses = [
  affine_map<(k, m, n, b, m2) -> (b, m, k, m2)>,
  affine_map<(k, m, n, b, m2) -> (n, k, b)>,
  affine_map<(k, m, n, b, m2) -> (n, b, m)>
]
#mkn_trait = {
  indexing_maps = #mkn_accesses,
  iterator_types = ["reduction", "parallel", "parallel", "parallel", "parallel"]
}

// CHECK-DAG: #[[$bmkm2_kkmm:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d3, d2, d1, d5, d6)>
// CHECK-DAG: #[[$nkb_kknn:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d2, d0, d5, d7)>
// CHECK-DAG: #[[$nbm_mmnn:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d0, d3, d6, d7)>

// CHECK-LABEL: @contraction_bmkm2_nkb_nbm(
func.func @contraction_bmkm2_nkb_nbm(%A : !A_bmkm2, %B : !B_nkb, %C : !C_nbm) -> !C_nbm {
  //      CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#[[$bmkm2_kkmm]], #[[$nkb_kknn]], #[[$nbm_mmnn]]],
  // CHECK-SAME:   ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel"]} 
  // CHECK-SAME:   ins(%{{.*}} : tensor<42x128x8x33x32x8xf32>, tensor<8x8x42x32x16xf32>)
  // CHECK-SAME:  outs(%{{.*}} : tensor<8x42x128x8x16xf32>)
  %0 = linalg.generic #mkn_trait ins(%A, %B : !A_bmkm2, !B_nkb) outs(%C : !C_nbm) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %d = arith.mulf %a, %b : f32
      %e = arith.addf %c, %d : f32
      linalg.yield %e : f32
  } -> !C_nbm
  return %0 : !C_nbm
}

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %func = transform.structured.match ops{["func.func"]} in %module_op
  %func_2 = transform.iree.pack_greedily %func
}
