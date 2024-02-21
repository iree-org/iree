// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-amdgpu-prepare-chained-matmul),canonicalize,cse)" %s | FileCheck %s

#accesses0 = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

#trait0 = {
  indexing_maps = #accesses0,
  iterator_types = ["parallel", "parallel", "reduction"]
}

builtin.module {
  // CHECK-DAG: #[[MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
  // CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
  // CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
  func.func @chained_matmul(%lhs : vector<32x8xf16>, %rhs : vector<16x8xf16>, %acc : vector<32x16xf16>,
    // CHECK: func.func @chained_matmul(%[[LHS:.*]]: vector<32x8xf16>, %[[RHS:.*]]: vector<16x8xf16>, %[[ACC:.*]]: vector<32x16xf16>
    // CHECK-SAME: %[[RHS2:.*]]: vector<8x16xf16>, %[[ACC2:.*]]: vector<32x8xf16>
    %rhs2 : vector<8x16xf16>, %acc2 : vector<32x8xf16>) -> vector<32x8xf16> {
    // CHECK: %[[TRANS_ACC:.*]] = vector.transpose %[[ACC]], [1, 0] : vector<32x16xf16> to vector<16x32xf16>
    // CHECK: %[[TRANS_RES:.*]] = vector.contract {indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
    // CHECK-SAME: %[[RHS]], %[[LHS]], %[[TRANS_ACC]] : vector<16x8xf16>, vector<32x8xf16> into vector<16x32xf16>
    // CHECK: %[[RES:.*]] = vector.transpose %[[TRANS_RES]], [1, 0] : vector<16x32xf16> to vector<32x16xf16>
    %result = vector.contract #trait0 %lhs, %rhs, %acc
      : vector<32x8xf16>, vector<16x8xf16> into vector<32x16xf16>
    // CHECK: %[[EXP:.*]] = math.exp2 %[[RES]] : vector<32x16xf16>
    %exp = math.exp2 %result : vector<32x16xf16>
    // CHECK: %[[TRANS_ACC2:.*]] = vector.transpose %[[ACC2]], [1, 0] : vector<32x8xf16> to vector<8x32xf16>
    // CHECK: %[[TRANS_RES2:.*]] = vector.contract {indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
    // CHECK-SAME: %[[RHS2]], %[[EXP]], %[[TRANS_ACC2]] : vector<8x16xf16>, vector<32x16xf16> into vector<8x32xf16>
    // CHECK: %[[RES2:.*]] = vector.transpose %[[TRANS_RES2]], [1, 0] : vector<8x32xf16> to vector<32x8xf16>
    %result2 = vector.contract #trait0 %exp, %rhs2, %acc2
      : vector<32x16xf16>, vector<8x16xf16> into vector<32x8xf16>
    func.return %result2 : vector<32x8xf16>
  }
}

// -----

#accesses0 = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

#trait0 = {
  indexing_maps = #accesses0,
  iterator_types = ["parallel", "parallel", "reduction"]
}

builtin.module {
  func.func @non_chained_matmul(%lhs : vector<32x8xf16>, %rhs : vector<16x8xf16>, %acc : vector<32x16xf16>
    // CHECK: func.func @non_chained_matmul(%[[LHS:.*]]: vector<32x8xf16>, %[[RHS:.*]]: vector<16x8xf16>, %[[ACC:.*]]: vector<32x16xf16>
    ) -> vector<32x16xf16> {
    // CHECK-NOT: vector.transpose
    %result = vector.contract #trait0 %lhs, %rhs, %acc
      : vector<32x8xf16>, vector<16x8xf16> into vector<32x16xf16>
    %exp = math.exp2 %result : vector<32x16xf16>
    func.return %exp : vector<32x16xf16>
  }
}

// -----

#accesses0 = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

#trait0 = {
  indexing_maps = #accesses0,
  iterator_types = ["parallel", "parallel", "reduction"]
}

builtin.module {
  func.func @chained_matmul_second_operand(%lhs : vector<32x8xf16>, %rhs : vector<16x8xf16>, %acc : vector<32x16xf16>,
    // CHECK: func.func @chained_matmul_second_operand(%[[LHS:.*]]: vector<32x8xf16>, %[[RHS:.*]]: vector<16x8xf16>, %[[ACC:.*]]: vector<32x16xf16>
    %lhs2 : vector<32x16xf16>, %acc2 : vector<32x32xf16>) -> vector<32x32xf16> {
    // CHECK-NOT: vector.transpose
    %result = vector.contract #trait0 %lhs, %rhs, %acc
      : vector<32x8xf16>, vector<16x8xf16> into vector<32x16xf16>
    %exp = math.exp2 %result : vector<32x16xf16>
    %result2 = vector.contract #trait0 %lhs2, %exp, %acc2
      : vector<32x16xf16>, vector<32x16xf16> into vector<32x32xf16>
    func.return %result2 : vector<32x32xf16>
  }
}

// -----

#accesses0 = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

#accesses1 = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]

#trait0 = {
  indexing_maps = #accesses0,
  iterator_types = ["parallel", "parallel", "reduction"]
}

#trait1 = {
  indexing_maps = #accesses1,
  iterator_types = ["parallel", "parallel", "reduction"]
}

builtin.module {
  func.func @chained_matmul_mmt_mm(%lhs : vector<32x8xf16>, %rhs : vector<16x8xf16>, %acc : vector<32x16xf16>,
    // CHECK-DAG: #[[MAP:.*]]  = affine_map<(d0, d1, d2) -> (d0, d2)>
    // CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
    // CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
    // CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
    // CHECK: func.func @chained_matmul_mmt_mm(%[[LHS:.*]]: vector<32x8xf16>, %[[RHS:.*]]: vector<16x8xf16>, %[[ACC:.*]]: vector<32x16xf16>
    // CHECK-SAME: %[[RHS2:.*]]: vector<16x8xf16>, %[[ACC2:.*]]: vector<32x8xf16>
    %rhs2 : vector<16x8xf16>, %acc2 : vector<32x8xf16>) -> vector<32x8xf16> {
    // CHECK: %[[TRANS_ACC:.*]] = vector.transpose %[[ACC]], [1, 0] : vector<32x16xf16> to vector<16x32xf16>
    // CHECK: %[[TRANS_RES:.*]] = vector.contract {indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
    // CHECK-SAME: %[[RHS]], %[[LHS]], %[[TRANS_ACC]] : vector<16x8xf16>, vector<32x8xf16> into vector<16x32xf16>
    // CHECK: %[[RES:.*]] = vector.transpose %[[TRANS_RES]], [1, 0] : vector<16x32xf16> to vector<32x16xf16>
    %result = vector.contract #trait0 %lhs, %rhs, %acc
      : vector<32x8xf16>, vector<16x8xf16> into vector<32x16xf16>
    // CHECK: %[[EXP:.*]] = math.exp2 %[[RES]] : vector<32x16xf16>
    %exp = math.exp2 %result : vector<32x16xf16>
    // CHECK: %[[TRANS_ACC2:.*]] = vector.transpose %[[ACC2]], [1, 0] : vector<32x8xf16> to vector<8x32xf16>
    // CHECK: %[[TRANS_EXP:.*]] = vector.transpose %[[EXP]], [1, 0] : vector<32x16xf16> to vector<16x32xf16>
    // CHECK: %[[TRANS_RHS2:.*]] = vector.transpose %[[RHS2]], [1, 0] : vector<16x8xf16> to vector<8x16xf16>
    // CHECK: %[[TRANS_RES2:.*]] = vector.contract {indexing_maps = [#[[MAP]], #[[MAP3]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
    // CHECK-SAME: %[[TRANS_RHS2]], %[[TRANS_EXP]], %[[TRANS_ACC2]] : vector<8x16xf16>, vector<16x32xf16> into vector<8x32xf16>
    // CHECK: %[[RES2:.*]] = vector.transpose %[[TRANS_RES2]], [1, 0] : vector<8x32xf16> to vector<32x8xf16>
    %result2 = vector.contract #trait1 %exp, %rhs2, %acc2
      : vector<32x16xf16>, vector<16x8xf16> into vector<32x8xf16>
    func.return %result2 : vector<32x8xf16>
  }
}
