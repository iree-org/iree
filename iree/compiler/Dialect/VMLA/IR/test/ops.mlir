// Tests the printing/parsing of the VMLA dialect ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @unaryOp
// CHECK-SAME: %[[SRC:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST:[a-zA-Z0-9$._-]+]]
func @unaryOp(%src : !vmla.buffer, %dst : !vmla.buffer) {
  // CHECK: vmla.log(%[[SRC]], %[[DST]]) : f32
  vmla.log(%src, %dst) : f32
  return
}

// -----

// CHECK-LABEL: @binaryOp
// CHECK-SAME: %[[LHS:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[RHS:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST:[a-zA-Z0-9$._-]+]]
func @binaryOp(%lhs : !vmla.buffer, %rhs : !vmla.buffer, %dst : !vmla.buffer) {
  // CHECK: vmla.atan2(%[[LHS]], %[[RHS]], %[[DST]]) : f32
  vmla.atan2(%lhs, %rhs, %dst) : f32
  return
}

// -----

// CHECK-LABEL: @ternaryOp
// CHECK-SAME: %[[A:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[B:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[C:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST:[a-zA-Z0-9$._-]+]]
func @ternaryOp(%a : !vmla.buffer, %b : !vmla.buffer, %c : !vmla.buffer,
                %dst : !vmla.buffer) {
  // CHECK: vmla.clamp(%[[A]], %[[B]], %[[C]], %[[DST]]) : f32
  vmla.clamp(%a, %b, %c, %dst) : f32
  return
}


// -----

// CHECK-LABEL: @vmla_convert
// CHECK-SAME: %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME: %[[DST:[a-zA-Z0-9]+]]
func @vmla_convert(%src : !vmla.buffer, %dst : !vmla.buffer) {
  // CHECK: vmla.convert(%[[SRC]], %[[DST]]) : f32 -> i8
  vmla.convert(%src, %dst) : f32 -> i8
  return
}

// -----

// CHECK-LABEL: @vmla_batch_matmul_pseudo
// CHECK-SAME: %[[LHS:[a-zA-Z0-9]+]]
// CHECK-SAME: %[[RHS:[a-zA-Z0-9]+]]
func @vmla_batch_matmul_pseudo(%lhs : tensor<32x256x128xf32>,
                               %rhs : tensor<32x1x128xf32>) {
  // CHECK: vmla.batch.matmul.pseudo(%[[LHS]], %[[RHS]]) :
  // CHECK-SAME: (tensor<32x256x128xf32>, tensor<32x1x128xf32>) ->
  // CHECK-SAME: tensor<32x1x256xf32>
  %dst = vmla.batch.matmul.pseudo(%lhs, %rhs) :
    (tensor<32x256x128xf32>, tensor<32x1x128xf32>) -> tensor<32x1x256xf32>
  return
}
