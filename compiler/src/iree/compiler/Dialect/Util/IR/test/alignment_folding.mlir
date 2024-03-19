// RUN: iree-opt --split-input-file --canonicalize --mlir-print-local-scope %s | iree-opt --split-input-file --mlir-print-local-scope | FileCheck %s

// CHECK-LABEL: @foldSameAlignment
// CHECK-SAME: (%[[VALUE:.+]]: index, %[[ALIGNMENT:.+]]: index)
util.func public @foldSameAlignment(%value: index, %alignment: index) -> index {
  // CHECK: %[[RET:.+]] = util.align %[[VALUE]], %[[ALIGNMENT]]
  %0 = util.align %value, %alignment : index
  // CHECK-NOT: util.align
  %1 = util.align %0, %alignment : index
  // CHECK: util.return %[[RET]]
  util.return %1 : index
}

// -----

// CHECK-LABEL: @foldGreaterAlignment
// CHECK-SAME: (%[[VALUE:.+]]: index)
util.func public @foldGreaterAlignment(%value: index) -> index {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  // CHECK: %[[RET:.+]] = util.align %[[VALUE]], %c16
  %0 = util.align %value, %c16 : index
  // CHECK-NOT: util.align
  %1 = util.align %0, %c8 : index
  // CHECK: util.return %[[RET]]
  util.return %1 : index
}

// -----

// CHECK-LABEL: @dontFoldLesserAlignment
// CHECK-SAME: (%[[VALUE:.+]]: index)
util.func public @dontFoldLesserAlignment(%value: index) -> index {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  // CHECK: %[[ALIGN16:.+]] = util.align %[[VALUE]], %c8
  %0 = util.align %value, %c8 : index
  // CHECK: %[[ALIGN8:.+]] = util.align %[[ALIGN16]], %c16
  %1 = util.align %0, %c16 : index
  // CHECK: util.return %[[ALIGN8]]
  util.return %1 : index
}

// -----

// CHECK-LABEL: @dontFoldMixedAlignment
// CHECK-SAME: (%[[VALUE:.+]]: index)
util.func public @dontFoldMixedAlignment(%value: index) -> index {
  %c9 = arith.constant 9 : index
  %c16 = arith.constant 16 : index
  // CHECK: %[[ALIGN16:.+]] = util.align %[[VALUE]], %c16
  %0 = util.align %value, %c16 : index
  // CHECK: %[[ALIGN9:.+]] = util.align %[[ALIGN16]], %c9
  %1 = util.align %0, %c9 : index
  // CHECK: util.return %[[ALIGN9]]
  util.return %1 : index
}

// -----

// CHECK-LABEL: @foldAlignmentRecursively
// CHECK-SAME: (%[[VALUE:.+]]: index, %[[ALIGNMENT:.+]]: index)
util.func public @foldAlignmentRecursively(%value: index, %alignment: index) -> index {
  %c16 = arith.constant 16 : index
  // CHECK: %[[ALIGN16:.+]] = util.align %[[VALUE]], %c16
  %0 = util.align %value, %c16 : index
  // CHECK: %[[ALIGN_DYNAMIC:.+]] = util.align %[[ALIGN16]], %[[ALIGNMENT]]
  %1 = util.align %0, %alignment : index
  // CHECK-NOT: util.align
  %2 = util.align %1, %c16 : index
  // CHECK: util.return %[[ALIGN_DYNAMIC]]
  util.return %2 : index
}

// -----

// CHECK-LABEL: @foldAddAlignment
// CHECK-SAME: (%[[LHS:.+]]: index, %[[RHS:.+]]: index, %[[ALIGNMENT:.+]]: index)
util.func public @foldAddAlignment(%lhs: index, %rhs: index, %alignment: index) -> index {
  // CHECK: %[[LHS_ALIGNED:.+]] = util.align %[[LHS]], %[[ALIGNMENT]]
  %lhs_aligned = util.align %lhs, %alignment : index
  // CHECK: %[[RHS_ALIGNED:.+]] = util.align %[[RHS]], %[[ALIGNMENT]]
  %rhs_aligned = util.align %rhs, %alignment : index
  // CHECK: %[[SUM_ALIGNED:.+]] = arith.addi %[[LHS_ALIGNED]], %[[RHS_ALIGNED]]
  %sum_aligned = arith.addi %lhs_aligned, %rhs_aligned : index
  // CHECK-NOT: util.align
  %result = util.align %sum_aligned, %alignment : index
  // CHECK: util.return %[[SUM_ALIGNED]]
  util.return %result : index
}

// -----

// CHECK-LABEL: @foldAddAlignmentConstant
// CHECK-SAME: (%[[LHS:.+]]: index)
util.func public @foldAddAlignmentConstant(%lhs: index) -> index {
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  // CHECK: %[[LHS_ALIGNED:.+]] = util.align %[[LHS]], %c64
  %lhs_aligned = util.align %lhs, %c64 : index
  // CHECK: %[[SUM_ALIGNED:.+]] = arith.addi %[[LHS_ALIGNED]], %c32
  %sum_aligned = arith.addi %lhs_aligned, %c32 : index
  // CHECK-NOT: util.align
  %result = util.align %sum_aligned, %c16 : index
  // CHECK: util.return %[[SUM_ALIGNED]]
  util.return %result : index
}

// -----

// CHECK-LABEL: @foldMulAlignmentConstant
// CHECK-SAME: (%[[LHS:.+]]: index)
util.func public @foldMulAlignmentConstant(%lhs: index) -> index {
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  // CHECK: %[[RESULT:.+]] = arith.muli %[[LHS]], %c2048
  %lhs_mul = arith.muli %lhs, %c2048 : index
  // CHECK-NOT: util.align
  %result = util.align %lhs_mul, %c64 : index
  // CHECK: util.return %[[RESULT]]
  util.return %result : index
}

// -----

// CHECK-LABEL: @foldConstantAlign
util.func public @foldConstantAlign() -> (index, index, index) {
  %c0 = arith.constant 0 : index
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index
  %c9 = arith.constant 9 : index
  %c64 = arith.constant 64 : index
  %0 = util.align %c0, %c64 : index
  %1 = util.align %c7, %c8 : index
  %2 = util.align %c9, %c8 : index
  // CHECK: util.return %c0, %c8, %c16
  util.return %0, %1, %2 : index, index, index
}

// -----

// CHECK-LABEL: @foldAffineAlign
util.func public @foldAffineAlign(%arg0: index) -> (index, index) {
  // CHECK: %[[A0:.+]] = affine.apply affine_map<()[s0] -> (s0 * 16384)>()[%arg0]
  %a0 = affine.apply affine_map<()[s0] -> (s0 * 16384)>()[%arg0]
  %c64 = arith.constant 64 : index
  %a1 = util.align %a0, %c64 : index
  // CHECK: %[[B0:.+]] = affine.apply affine_map<()[s0] -> ((s0 * s0) * 4)>()[%arg0]
  %b0 = affine.apply affine_map<()[s0] -> ((s0 * s0) * 4)>()[%arg0]
  %c4 = arith.constant 4 : index
  %b1 = util.align %b0, %c4 : index
  // CHECK: util.return %[[A0]], %[[B0]]
  util.return %a1, %b1 : index, index
}

// -----

// CHECK-LABEL: @sizeofWholeInt
util.func public @sizeofWholeInt() -> index {
  // CHECK: = arith.constant 4 : index
  %0 = util.sizeof i32
  util.return %0 : index
}

// -----

// CHECK-LABEL: @sizeofSubByteInt
util.func public @sizeofSubByteInt() -> index {
  // CHECK: = arith.constant 2 : index
  %0 = util.sizeof i12
  util.return %0 : index
}

// -----

// CHECK-LABEL: @sizeofFloat
util.func public @sizeofFloat() -> index {
  // CHECK: = arith.constant 4 : index
  %0 = util.sizeof f32
  util.return %0 : index
}
