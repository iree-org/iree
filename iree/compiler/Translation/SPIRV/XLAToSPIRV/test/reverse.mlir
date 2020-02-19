// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv -verify-diagnostics -o - %s | IreeFileCheck %s

module {
  // CHECK: spv.func @reverse_2d
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]*]]: !spv.ptr<!spv.struct<!spv.array<144 x f32 [4]> [0]>, StorageBuffer>
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]*]]: !spv.ptr<!spv.struct<!spv.array<144 x f32 [4]> [0]>, StorageBuffer>
  func @reverse_2d(%arg0: memref<12x12xf32>, %arg1 : memref<12x12xf32>)
  attributes  {iree.executable.export, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0]]
    // CHECK: [[VAL:%.*]]  = spv.Load "StorageBuffer" [[ARG0LOADPTR]] : f32
    %0 = iree.load_input(%arg0 : memref<12x12xf32>) : tensor<12x12xf32>
    %1 = "xla_hlo.reverse"(%0) {dimensions = dense<[1, 0]> : tensor<2xi64>} : (tensor<12x12xf32>) -> tensor<12x12xf32>
    // CHECK: [[ARG1STOREPTR:%.*]] = spv.AccessChain [[ARG1]]
    // CHECK: spv.Store "StorageBuffer" [[ARG1STOREPTR]], [[VAL]]
    iree.store_output(%1 : tensor<12x12xf32>, %arg1 : memref<12x12xf32>)
    iree.return
  }
}

// -----

module {
  // CHECK: spv.func @reverse_3d
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]*]]: !spv.ptr<!spv.struct<!spv.array<27 x f32 [4]> [0]>, StorageBuffer>
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]*]]: !spv.ptr<!spv.struct<!spv.array<27 x f32 [4]> [0]>, StorageBuffer>
  func @reverse_3d(%arg0: memref<3x3x3xf32>, %arg1 : memref<3x3x3xf32>)
  attributes  {iree.executable.export, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0]]
    // CHECK: [[VAL:%.*]]  = spv.Load "StorageBuffer" [[ARG0LOADPTR]] : f32
    %0 = iree.load_input(%arg0 : memref<3x3x3xf32>) : tensor<3x3x3xf32>
    %1 = "xla_hlo.reverse"(%0) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<3x3x3xf32>) -> tensor<3x3x3xf32>
    // CHECK: [[ARG1STOREPTR:%.*]] = spv.AccessChain [[ARG1]]
    // CHECK: spv.Store "StorageBuffer" [[ARG1STOREPTR]], [[VAL]]
    iree.store_output(%1 : tensor<3x3x3xf32>, %arg1 : memref<3x3x3xf32>)
    iree.return
  }
}
