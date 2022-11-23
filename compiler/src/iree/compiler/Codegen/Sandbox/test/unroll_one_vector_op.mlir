// RUN: iree-opt %s --pass-pipeline='builtin.module(func.func(unroll-one-vector-op{anchor-func=test anchor-op=vector.contract source-shape=4,4,3 target-shape=2,4,3}))' | FileCheck %s

#matmul_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
#matmul_trait = {
  indexing_maps = #matmul_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func.func @test
func.func @test(%a: vector<4x3xf32>, %b: vector<3x4xf32>, %c: vector<4x4xf32>) -> vector<4x4xf32> {
  // CHECK: vector.contract {{.*}} : vector<2x3xf32>, vector<3x4xf32> into vector<2x4xf32>
  // CHECK: vector.contract {{.*}} : vector<2x3xf32>, vector<3x4xf32> into vector<2x4xf32>
  %d = vector.contract #matmul_trait %a, %b, %c: vector<4x3xf32>, vector<3x4xf32> into vector<4x4xf32>
  return %d: vector<4x4xf32>
}
