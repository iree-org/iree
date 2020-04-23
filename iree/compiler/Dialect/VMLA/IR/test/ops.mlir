// Tests the printing/parsing of the VMLA dialect ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @unaryOp
// CHECK-SAME: %[[SRC:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST:[a-zA-Z0-9$._-]+]]
func @unaryOp(%src : !vmla.buffer, %dst : !vmla.buffer) {
  // CHECK: vmla.log %[[SRC]], out %[[DST]] : f32
  vmla.log %src, out %dst : f32
  return
}

// -----

// CHECK-LABEL: @binaryOp
// CHECK-SAME: %[[LHS:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[RHS:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST:[a-zA-Z0-9$._-]+]]
func @binaryOp(%lhs : !vmla.buffer, %rhs : !vmla.buffer, %dst : !vmla.buffer) {
  // CHECK: vmla.atan2 %[[LHS]], %[[RHS]], out %[[DST]] : f32
  vmla.atan2 %lhs, %rhs, out %dst : f32
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
  // CHECK: vmla.clamp %[[A]], %[[B]], %[[C]], out %[[DST]] : f32
  vmla.clamp %a, %b, %c, out %dst : f32
  return
}


// -----

// CHECK-LABEL: @vmla_convert
// CHECK-SAME: %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME: %[[DST:[a-zA-Z0-9]+]]
func @vmla_convert(%src : !vmla.buffer, %dst : !vmla.buffer) {
  // CHECK: vmla.convert %[[SRC]], out %[[DST]] : f32 -> i8
  vmla.convert %src, out %dst : f32 -> i8
  return
}

// -----

// CHECK-LABEL: @vmla_batch_matmul_pseudo
// CHECK-SAME: %[[LHS:[a-zA-Z0-9]+]]
// CHECK-SAME: %[[RHS:[a-zA-Z0-9]+]]
func @vmla_batch_matmul_pseudo(%lhs : tensor<32x256x128xf32>,
                               %rhs : tensor<32x1x128xf32>) {
  // CHECK: vmla.batch.matmul.pseudo %[[LHS]], %[[RHS]] :
  // CHECK-SAME: (tensor<32x256x128xf32>, tensor<32x1x128xf32>) ->
  // CHECK-SAME: tensor<32x1x256xf32>
  %dst = vmla.batch.matmul.pseudo %lhs, %rhs :
    (tensor<32x256x128xf32>, tensor<32x1x128xf32>) -> tensor<32x1x256xf32>
  return
}

// -----

// CHECK-LABEL: @vmla_batch_matmul
// CHECK-SAME: %[[LHS:[a-zA-Z0-9]+]]
// CHECK-SAME: %[[LHS_SHAPE:[a-zA-Z0-9]+]]
// CHECK-SAME: %[[RHS:[a-zA-Z0-9]+]]
// CHECK-SAME: %[[RHS_SHAPE:[a-zA-Z0-9]+]]
// CHECK-SAME: %[[DST:[a-zA-Z0-9]+]]
// CHECK-SAME: %[[DST_SHAPE:[a-zA-Z0-9]+]]
func @vmla_batch_matmul(%lhs : !vmla.buffer,
                        %lhs_shape : !shapex.ranked_shape<[8,4,4]>,
                        %rhs : !vmla.buffer,
                        %rhs_shape : !shapex.ranked_shape<[8,1,4]>,
                        %dst : !vmla.buffer,
                        %dst_shape : !shapex.ranked_shape<[8,1,4]>) {
  // CHECK:      vmla.batch.matmul
  // CHECK-SAME: %[[LHS]](%[[LHS_SHAPE]] : !shapex.ranked_shape<[8,4,4]>) : f32,
  // CHECK-SAME: %[[RHS]](%[[RHS_SHAPE]] : !shapex.ranked_shape<[8,1,4]>) : f32,
  // CHECK-SAME: out
  // CHECK-SAME: %[[DST]](%[[DST_SHAPE]] : !shapex.ranked_shape<[8,1,4]>) : f32
  vmla.batch.matmul %lhs(%lhs_shape : !shapex.ranked_shape<[8,4,4]>) : f32,
                    %rhs(%rhs_shape : !shapex.ranked_shape<[8,1,4]>) : f32,
                    out %dst(%dst_shape : !shapex.ranked_shape<[8,1,4]>) : f32
  return
}

// -----

// CHECK-LABEL: @vmla_transpose
// CHECK-SAME: %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME: %[[SRC_SHAPE:[a-zA-Z0-9]+]]
// CHECK-SAME: %[[DST:[a-zA-Z0-9]+]]
// CHECK-SAME: %[[DST_SHAPE:[a-zA-Z0-9]+]]
func @vmla_transpose(%src : !vmla.buffer,
                     %src_shape : !shapex.ranked_shape<[64,32,32,10]>,
                     %dst : !vmla.buffer,
                     %dst_shape : !shapex.ranked_shape<[64,10,32,32]>) {
  // CHECK:      vmla.transpose
  // CHECK-SAME: %[[SRC]](%[[SRC_SHAPE]] : !shapex.ranked_shape<[64,32,32,10]>),
  // CHECK-SAME: out
  // CHECK-SAME: %[[DST]](%[[DST_SHAPE]] : !shapex.ranked_shape<[64,10,32,32]>)
  // CHECK-SAME: {permutation = dense<[0, 3, 2, 1]> : tensor<4xi32>} : f32
  vmla.transpose %src(%src_shape : !shapex.ranked_shape<[64,32,32,10]>),
                 out %dst(%dst_shape : !shapex.ranked_shape<[64,10,32,32]>)
                 {permutation = dense<[0, 3, 2, 1]> : tensor<4xi32>} : f32
  return
}
