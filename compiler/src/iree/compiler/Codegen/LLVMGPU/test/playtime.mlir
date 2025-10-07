// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-llvmgpu-vector-to-gpu))" --debug --split-input-file | FileCheck %s

// Checking a failed (flakey?) test in vector_to_gpu.mlir, fails on windows

// CHECK-LABEL: func.func @ksplitmatmul_basic
// CHECK: vector.broadcast
// CHECK: vector.transpose
// CHECK: return
func.func @ksplitmatmul_basic(%a: memref<128x16x256xf32>) -> vector<16x1x8xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %0 = vector.transfer_read %a[%c2, %c3, %c4], %cst {in_bounds = [true, true, true]} : memref<128x16x256xf32>, vector<16x1x8xf32>
  return %0 : vector<16x1x8xf32>
}
