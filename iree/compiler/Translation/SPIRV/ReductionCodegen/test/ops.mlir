// RUN: iree-opt -iree-spirv-reduction-fn-lowering -o - %s | IreeFileCheck %s

// CHECK-LABEL: func @reduction_max_apply
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: i32
// CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]: !spv.ptr<i32, StorageBuffer>
func @reduction_max_apply(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  // CHECK: spv.AtomicSMax "Device" "AcquireRelease" [[ARG1]], [[ARG0]]
  %0 = xla_hlo.max %arg0, %arg1 : tensor<i32>
  iree.return %0 : tensor<i32>
}

// CHECK-LABEL: func @reduction_min_apply
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: i32
// CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]: !spv.ptr<i32, StorageBuffer>
func @reduction_min_apply(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  // CHECK: spv.AtomicSMin "Device" "AcquireRelease" [[ARG1]], [[ARG0]]
  %0 = xla_hlo.min %arg0, %arg1 : tensor<i32>
  iree.return %0 : tensor<i32>
}

// CHECK-LABEL: func @reduction_iadd_apply
func @reduction_iadd_apply(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  // CHECK: spv.AtomicIAdd
  %0 = std.addi %arg0, %arg1 : tensor<i32>
  iree.return %0 : tensor<i32>
}

