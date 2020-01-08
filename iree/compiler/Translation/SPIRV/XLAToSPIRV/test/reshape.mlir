// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv -verify-diagnostics -o - %s | IreeFileCheck %s

module {
  // CHECK: func @reshape_2D_2D
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<504 x i32 [4]> [0]>, StorageBuffer>
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<504 x i32 [4]> [0]>, StorageBuffer>
  func @reshape_2D_2D(%arg0: memref<24x21xi32>, %arg1: memref<12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0]]
    // CHECK: [[VAL:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]]
    %0 = iree.load_input(%arg0 : memref<24x21xi32>) : tensor<24x21xi32>
    %1 = "xla_hlo.reshape"(%0) : (tensor<24x21xi32>) -> tensor<12x42xi32>
    // CHECK: [[ARG1STOREPTR:%.*]] = spv.AccessChain [[ARG1]]
    // CHECK: spv.Store "StorageBuffer" [[ARG1STOREPTR]], [[VAL]]
    iree.store_output(%1 : tensor<12x42xi32>, %arg1 : memref<12x42xi32>)
    iree.return
  }
}

// -----

module {
  // CHECK: func @reshape_3D_2D
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<504 x i32 [4]> [0]>, StorageBuffer>
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<504 x i32 [4]> [0]>, StorageBuffer>
  func @reshape_3D_2D(%arg0: memref<4x6x21xi32>, %arg1: memref<12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0]]
    // CHECK: [[VAL:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]]
    %0 = iree.load_input(%arg0 : memref<4x6x21xi32>) : tensor<4x6x21xi32>
    %1 = "xla_hlo.reshape"(%0) : (tensor<4x6x21xi32>) -> tensor<12x42xi32>
    // CHECK: [[ARG1STOREPTR:%.*]] = spv.AccessChain [[ARG1]]
    // CHECK: spv.Store "StorageBuffer" [[ARG1STOREPTR]], [[VAL]]
    iree.store_output(%1 : tensor<12x42xi32>, %arg1 : memref<12x42xi32>)
    iree.return
  }
}

// -----

module {
  // CHECK: func @reshape_2D_3D
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<504 x i32 [4]> [0]>, StorageBuffer>
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<504 x i32 [4]> [0]>, StorageBuffer>
  func @reshape_2D_3D(%arg0: memref<24x21xi32>, %arg1: memref<12x6x7xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0]]
    // CHECK: [[VAL:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]]
    %0 = iree.load_input(%arg0 : memref<24x21xi32>) : tensor<24x21xi32>
    %1 = "xla_hlo.reshape"(%0) : (tensor<24x21xi32>) -> tensor<12x6x7xi32>
    // CHECK: [[ARG1STOREPTR:%.*]] = spv.AccessChain [[ARG1]]
    // CHECK: spv.Store "StorageBuffer" [[ARG1STOREPTR]], [[VAL]]
    iree.store_output(%1 : tensor<12x6x7xi32>, %arg1 : memref<12x6x7xi32>)
    iree.return
  }
}
