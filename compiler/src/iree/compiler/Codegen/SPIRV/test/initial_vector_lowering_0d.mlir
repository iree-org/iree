// RUN: iree-opt %s --pass-pipeline='builtin.module(func.func(iree-spirv-initial-vector-lowering))' | FileCheck %s

// Make sure that we handle 0-D vector types correctly during unrolling in
// initial vector lowering.

// CHECK-LABEL: func.func @main
// CHECK: scf.for
// CHECK:   vector.transfer_read {{.+}} : tensor<f32>, vector<f32>
// CHECK:   arith.mulf {{.+}} : vector<f32>
// CHECK:   %[[VEC:.+]] = vector.extract {{.+}}[] : f32 from vector<f32>
// CHECK:   vector.insert %[[VEC]], {{.+}} [0] : f32 into vector<1xf32>
// CHECK:   vector.transfer_write {{.+}} : vector<1xf32>, tensor<32xf32>

func.func @main(%0: tensor<32xf32>, %1: tensor<f32>) -> tensor<32xf32> {
  %cst = arith.constant dense<1.500000e+01> : vector<f32>
  %poison = ub.poison : f32
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %2 = scf.for %arg0 = %c0 to %c32 step %c1 iter_args(%arg1 = %0) -> (tensor<32xf32>) {
    %3 = vector.transfer_read %1[], %poison : tensor<f32>, vector<f32>
    %4 = arith.mulf %3, %cst : vector<f32>
    %5 = vector.broadcast %4 : vector<f32> to vector<1xf32>
    %6 = vector.transfer_write %5, %arg1[%arg0] {in_bounds = [true]} : vector<1xf32>, tensor<32xf32>
    scf.yield %6 : tensor<32xf32>
  }
  return %2 : tensor<32xf32>
}
