// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv -verify-diagnostics -o - %s | IreeFileCheck %s

module {
  // CHECK: func @slice_unit_stride
  // CHECK-SAME: [[ARG0:%.*]]: !spv.ptr<!spv.struct<!spv.array<36 x f32 [4]> [0]>, StorageBuffer>
  // CHECK-SAME: [[ARG1:%.*]]: !spv.ptr<!spv.struct<!spv.array<6 x f32 [4]> [0]>, StorageBuffer>
  func @slice_unit_stride(%arg0: memref<6x6xf32>, %arg1: memref<2x3xf32>)
  attributes {iree.executable.export, iree.executable.workload = dense<[6, 1]> : tensor<2xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0]]
    // CHECK: [[VAL0:%.*]]  = spv.Load "StorageBuffer" [[ARG0LOADPTR]] : f32
    // CHECK: [[ARG1STOREPTR:%.*]] = spv.AccessChain [[ARG1]]
    // CHECK: spv.Store "StorageBuffer" [[ARG1STOREPTR]], [[VAL0]] : f32
    %0 = iree.load_input(%arg0 : memref<6x6xf32>) : tensor<6x6xf32>
    %1 = "xla_hlo.slice"(%0) {start_indices = dense<[2, 1]> : tensor<2xi64>, limit_indices = dense<[4, 4]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} : (tensor<6x6xf32>) -> tensor<2x3xf32>
    iree.store_output(%1 : tensor<2x3xf32>, %arg1 : memref<2x3xf32>)
    iree.return
  }
}

// -----

module {
  // CHECK: func @slice_non_unit_stride
  // CHECK-SAME: [[ARG0:%.*]]: !spv.ptr<!spv.struct<!spv.array<36 x f32 [4]> [0]>, StorageBuffer>
  // CHECK-SAME: [[ARG1:%.*]]: !spv.ptr<!spv.struct<!spv.array<6 x f32 [4]> [0]>, StorageBuffer>
  func @slice_non_unit_stride(%arg0: memref<6x6xf32>, %arg1: memref<2x3xf32>)
  attributes {iree.executable.export, iree.executable.workload = dense<[6, 1]> : tensor<2xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0]]
    // CHECK: [[VAL0:%.*]]  = spv.Load "StorageBuffer" [[ARG0LOADPTR]] : f32
    // CHECK: [[ARG1STOREPTR:%.*]] = spv.AccessChain [[ARG1]]
    // CHECK: spv.Store "StorageBuffer" [[ARG1STOREPTR]], [[VAL0]] : f32
    %0 = iree.load_input(%arg0 : memref<6x6xf32>) : tensor<6x6xf32>
    %1 = "xla_hlo.slice"(%0) {start_indices = dense<[2, 1]> : tensor<2xi64>, limit_indices = dense<[4, 6]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<6x6xf32>) -> tensor<2x3xf32>
    iree.store_output(%1 : tensor<2x3xf32>, %arg1 : memref<2x3xf32>)
    iree.return
  }
}
