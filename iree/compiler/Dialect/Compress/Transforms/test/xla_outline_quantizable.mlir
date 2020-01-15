// RUN: iree-opt -split-input-file -verify-diagnostics -iree-compress-xla-outline-quantizable %s | IreeFileCheck %s

// CHECK-LABEL: @add
func @add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
  // CHECK: iree_compress.quant_region
  // CHECK: xla_hlo.add
  // CHECK: logical_kernel = "BINARY_ADD"
  %0 = xla_hlo.add %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @add_broadcast_rejected
func @add_broadcast_rejected(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
  // CHECK-NOT: iree_compress.quant_region
  %0 = xla_hlo.add %arg0, %arg1 {broadcast_dimensions = dense<[]> : tensor<0xi64>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @div
func @div(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
  // CHECK: iree_compress.quant_region
  // CHECK: xla_hlo.div
  // CHECK: logical_kernel = "BINARY_DIV"
  %0 = xla_hlo.div %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @div_broadcast_rejected
func @div_broadcast_rejected(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
  // CHECK-NOT: iree_compress.quant_region
  %0 = xla_hlo.div %arg0, %arg1 {broadcast_dimensions = dense<[]> : tensor<0xi64>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @mul
func @mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
  // CHECK: iree_compress.quant_region
  // CHECK: xla_hlo.mul
  // CHECK: logical_kernel = "BINARY_MUL"
  %0 = xla_hlo.mul %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @mul_broadcast_rejected
func @mul_broadcast_rejected(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
  // CHECK-NOT: iree_compress.quant_region
  %0 = xla_hlo.mul %arg0, %arg1 {broadcast_dimensions = dense<[]> : tensor<0xi64>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @sub
func @sub(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
  // CHECK: iree_compress.quant_region
  // CHECK: xla_hlo.sub
  // CHECK: logical_kernel = "BINARY_SUB"
  %0 = xla_hlo.sub %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @sub_broadcast_rejected
func @sub_broadcast_rejected(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
  // CHECK-NOT: iree_compress.quant_region
  %0 = xla_hlo.sub %arg0, %arg1 {broadcast_dimensions = dense<[]> : tensor<0xi64>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @select
func @select(%arg0: tensor<4xi1>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> (tensor<4xf32>) {
  // TODO: CHECK ME
  %0 = "xla_hlo.select"(%arg0, %arg1, %arg2) : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @copy
func @copy(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
  // TODO: CHECK ME
  %0 = "xla_hlo.copy"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

