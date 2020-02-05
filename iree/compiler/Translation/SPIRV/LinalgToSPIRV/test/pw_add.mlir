// RUN: iree-opt -pass-pipeline='iree-xla-to-linalg-to-spirv' %s | IreeFileCheck %s

module {
  func @simple_load_store(%arg0: memref<4x8xi32>, %arg1: memref<4x8xi32>, %arg2 : memref<4x8xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[8, 4, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[2, 2, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: spv.module
    %0 = iree.load_input(%arg0 : memref<4x8xi32>) : tensor<4x8xi32>
    %1 = iree.load_input(%arg1 : memref<4x8xi32>) : tensor<4x8xi32>
    %2 = "xla_hlo.add"(%0, %1) : (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
    iree.store_output(%2 : tensor<4x8xi32>, %arg2 : memref<4x8xi32>)
    iree.return
  }
}
