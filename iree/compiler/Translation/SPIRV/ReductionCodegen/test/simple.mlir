// RUN: iree-opt -iree-spirv-prepare-reduction-dispatch -verify-diagnostics -o - %s  | IreeFileCheck %s

module {
  // CHECK-LABEL: func @reduction_entry
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]*]]: memref<5x4xi32>
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]*]]: memref<i32>
  // CHECK-SAME: [[ARG2:%[a-zA-Z0-9]*]]: memref<4xi32> {iree.executable.reduction.output}
  // CHECK-SAME: iree.executable.reduction.apply = [[APPLYFN:@[a-zA-Z0-9_]*]]
  func @reduction_entry(memref<5x4xi32>, memref<i32>, memref<4xi32>) attributes {iree.executable.export, iree.executable.reduction, iree.executable.reduction.apply = @reduction_apply, iree.executable.reduction.dimension = 0 : i32, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi64>, iree.executable.workload = dense<[4, 5, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32}
  // CHECK: [[TENSOR:%.*]] = iree.load_input([[ARG0]] : memref<5x4xi32>)  : tensor<5x4xi32>
  // CHECK: iree.store_reduce([[TENSOR]] : tensor<5x4xi32>, [[ARG2]] : memref<4xi32>, [[APPLYFN]])

  func @reduction_apply(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = xla_hlo.max %arg0, %arg1 : tensor<i32>
    iree.return %0 : tensor<i32>
  }
}

