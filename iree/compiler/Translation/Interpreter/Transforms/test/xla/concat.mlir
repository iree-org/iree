// RUN: iree-opt --lower-xla-to-iree-interpreter %s --split-input-file | IreeFileCheck %s

// CHECK-LABEL: func @concat.1D
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
func @concat.1D(%arg0 : tensor<4xi32>, %arg1 : tensor<4xi32>) -> tensor<8xi32> {
  // CHECK-DAG: [[ARG0_MEMREF:%.+]] = iree_interp.tensor_to_memref([[ARG0]]
  // CHECK-DAG: [[ARG1_MEMREF:%.+]] = iree_interp.tensor_to_memref([[ARG1]]
  // CHECK:     [[RES:%.+]] = "iree_hl_interp.concat"([[ARG0_MEMREF]], [[ARG1_MEMREF]]) {dimension = 0 : i32}
  %0 = "xla_hlo.concatenate"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<4xi32>, tensor<4xi32>) -> tensor<8xi32>

  // CHECK: [[RES_TENSOR:%.+]] = iree_interp.memref_to_tensor([[RES]]
  // CHECK: return [[RES_TENSOR]]
  return %0 : tensor<8xi32>
}

// -----

// CHECK-LABEL: func @concat.2D.Dim0
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
func @concat.2D.Dim0(%arg0 : tensor<4x4xi32>, %arg1 : tensor<4x4xi32>) -> tensor<8x4xi32> {
  // CHECK-DAG: [[ARG0_MEMREF:%.+]]  = iree_interp.tensor_to_memref([[ARG0]]
  // CHECK-DAG: [[ARG1_MEMREF:%.+]]  = iree_interp.tensor_to_memref([[ARG1]]
  // CHECK:     [[RES:%.+]] = "iree_hl_interp.concat"([[ARG0_MEMREF]], [[ARG1_MEMREF]]) {dimension = 0 : i32}
  %0 = "xla_hlo.concatenate"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<8x4xi32>

  // CHECK: [[RES_TENSOR:%.+]] = iree_interp.memref_to_tensor([[RES]]
  // CHECK: return [[RES_TENSOR]]
  return %0 : tensor<8x4xi32>
}

// -----

// CHECK-LABEL: func @concat.2D.Dim1
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
func @concat.2D.Dim1(%arg0 : tensor<4x4xi32>, %arg1 : tensor<4x4xi32>) -> tensor<4x8xi32> {
  // CHECK-DAG: [[ARG0_MEMREF:%.+]]  = iree_interp.tensor_to_memref([[ARG0]]
  // CHECK-DAG: [[ARG1_MEMREF:%.+]]  = iree_interp.tensor_to_memref([[ARG1]]
  // CHECK:     [[RES:%.+]] = "iree_hl_interp.concat"([[ARG0_MEMREF]], [[ARG1_MEMREF]]) {dimension = 1 : i32}
  %0 = "xla_hlo.concatenate"(%arg0, %arg1) {dimension = 1 : i64} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x8xi32>

  // CHECK: [[RES_TENSOR:%.+]] = iree_interp.memref_to_tensor([[RES]]
  // CHECK: return [[RES_TENSOR]]
  return %0 : tensor<4x8xi32>
}
