// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmgpu-vector-lowering))" --split-input-file %s | FileCheck %s

#lhs = affine_map<(p, j, k) -> (p, j, k)>
#rhs = affine_map<(p, j, k) -> (p, k, j)>
#out = affine_map<(p, j, k) -> (p)>

// CHECK-LABEL: func.func @permuted_equal_reductions
// CHECK-SAME: %[[LHS:.*]]: vector<1x2x2xf32>, %[[RHS:.*]]: vector<1x2x2xf32>
// CHECK: vector.transpose %[[LHS]], [1, 2, 0]
// CHECK: vector.transpose %[[RHS]], [2, 1, 0]
func.func @permuted_equal_reductions(%a: vector<1x2x2xf32>, %b: vector<1x2x2xf32>, %acc: vector<1xf32>) -> vector<1xf32> {
  %r = vector.contract {indexing_maps = [#lhs, #rhs, #out], iterator_types = ["parallel", "reduction", "reduction"], kind = #vector.kind<add>} %a, %b, %acc : vector<1x2x2xf32>, vector<1x2x2xf32> into vector<1xf32>
  return %r : vector<1xf32>
}

// -----

#lhs = affine_map<(p, j, k) -> (p, j, k)>
#rhs = affine_map<(p, j, k) -> (k, p, j)>
#out = affine_map<(p, j, k) -> (p)>

// CHECK-LABEL: func.func @permuted_unequal_reductions
// CHECK-SAME: %[[LHS:.*]]: vector<2x3x4xf32>, %[[RHS:.*]]: vector<4x2x3xf32>
// CHECK: vector.transpose %[[LHS]], [1, 2, 0] : vector<2x3x4xf32> to vector<3x4x2xf32>
// CHECK: vector.transpose %[[RHS]], [2, 0, 1] : vector<4x2x3xf32> to vector<3x4x2xf32>
func.func @permuted_unequal_reductions(%a: vector<2x3x4xf32>, %b: vector<4x2x3xf32>, %acc: vector<2xf32>) -> vector<2xf32> {
  %r = vector.contract {indexing_maps = [#lhs, #rhs, #out], iterator_types = ["parallel", "reduction", "reduction"], kind = #vector.kind<add>} %a, %b, %acc : vector<2x3x4xf32>, vector<4x2x3xf32> into vector<2xf32>
  return %r : vector<2xf32>
}
