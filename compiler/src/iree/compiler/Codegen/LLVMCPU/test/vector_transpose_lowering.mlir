// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-llvmcpu-vector-transpose-lowering))' --split-input-file %s | FileCheck %s

// Verify that the vector transpose lowering patterns trigger as expected. We
// shouldn't check the pattern output in detail as that testing should happen in
// MLIR, where the patterns are implemented.

func.func @i4_transpose(%a: vector<8x16xi4>) -> vector<16x8xi4> {
  %0 = vector.transpose %a, [1, 0] : vector<8x16xi4> to vector<16x8xi4>
  return %0 : vector<16x8xi4>
}

// CHECK-LABEL: func.func @i4_transpose(
//       CHECK:   arith.extsi %{{.*}} : vector<8x16xi4> to vector<8x16xi8>
//       CHECK:   vector.shuffle
//       CHECK:   arith.trunci %{{.*}} : vector<16x8xi8> to vector<16x8xi4>

