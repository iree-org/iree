// Tests the printing/parsing of the VMLA dialect ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @unaryOp
// CHECK-SAME: %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME: %[[DST:[a-zA-Z0-9]+]]
func @unaryOp(%src : !vmla.buffer, %dst : !vmla.buffer) {
  // CHECK: vmla.log(%[[SRC]], %[[DST]]) : !vmla.buffer f32
  vmla.log(%src, %dst) : !vmla.buffer f32
  return
}

// -----

// CHECK-LABEL: @binaryOp
// CHECK-SAME: %[[LHS:[a-zA-Z0-9]+]]
// CHECK-SAME: %[[RHS:[a-zA-Z0-9]+]]
// CHECK-SAME: %[[DST:[a-zA-Z0-9]+]]
func @binaryOp(%lhs : !vmla.buffer, %rhs : !vmla.buffer, %dst : !vmla.buffer) {
  // CHECK: vmla.atan2(%[[LHS]], %[[RHS]], %[[DST]]) : !vmla.buffer f32
  vmla.atan2(%lhs, %rhs, %dst) : !vmla.buffer f32
  return
}

// -----

// CHECK-LABEL: @ternaryOp
// CHECK-SAME: %[[A:[a-zA-Z0-9]+]]
// CHECK-SAME: %[[B:[a-zA-Z0-9]+]]
// CHECK-SAME: %[[C:[a-zA-Z0-9]+]]
// CHECK-SAME: %[[DST:[a-zA-Z0-9]+]]
func @ternaryOp(%a : !vmla.buffer, %b : !vmla.buffer, %c : !vmla.buffer, 
                %dst : !vmla.buffer) {
  // CHECK: vmla.clamp(%[[A]], %[[B]], %[[C]], %[[DST]]) : !vmla.buffer f32
  vmla.clamp(%a, %b, %c, %dst) : !vmla.buffer f32
  return
}