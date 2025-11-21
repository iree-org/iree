// RUN: iree-opt %s --pass-pipeline='builtin.module(func.func(iree-spirv-initial-vector-lowering))' | FileCheck %s

// Make sure that we handle 0-D vector types correctly during unrolling in
// initial vector lowering.

// CHECK-LABEL: func.func @main
// CHECK: scf.for
// CHECK:   vector.transfer_read {{.+}} : tensor<f32>, vector<f32>
// CHECK:   arith.mulf {{.+}} : vector<f32>
// CHECK:   vector.broadcast {{.+}} : vector<f32> to vector<1xf32>
// CHECK:   vector.transfer_write {{.+}} : vector<1xf32>, tensor<32xf32>

func.func @main(%4: tensor<32xf32>, %5: tensor<f32>) -> tensor<32xf32> {
  %cst = arith.constant dense<1.500000e+01> : vector<f32>
  %0 = ub.poison : f32
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %6 = scf.for %arg0 = %c0 to %c32 step %c1 iter_args(%arg1 = %4) -> (tensor<32xf32>) {
    %7 = vector.transfer_read %5[], %0 : tensor<f32>, vector<f32>
    %8 = arith.mulf %7, %cst : vector<f32>
    %10 = vector.broadcast %8 : vector<f32> to vector<1xf32>
    %11 = vector.transfer_write %10, %arg1[%arg0] {in_bounds = [true]} : vector<1xf32>, tensor<32xf32>
    scf.yield %11 : tensor<32xf32>
  }
  return %6 : tensor<32xf32>
}
