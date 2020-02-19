// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv -verify-diagnostics -o - %s | IreeFileCheck %s

module {
  // CHECK: spv.func @pad_zero_interior
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<48 x f32 [4]> [0]>, StorageBuffer>
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<216 x f32 [4]> [0]>, StorageBuffer>
  func @pad_zero_interior(%arg0 : memref<12x4xf32>, %arg1 : memref<18x12xf32>)
  attributes  {iree.executable.export, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0]]
    // CHECK: [[INPUTVAL:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]] : f32
    %0 = iree.load_input(%arg0 : memref<12x4xf32>) : tensor<12x4xf32>

    // CHECK: [[PADVAL:%.*]] = spv.constant 0.000000e+00 : f32
    %1 = constant dense<0.0> : tensor<f32>

    // CHECK: [[VALUE:%.*]] = spv.Select {{%.*}}, [[INPUTVAL]], [[PADVAL]] : i1, f32
    %2 = "xla_hlo.pad"(%0, %1) {edge_padding_high = dense<[2, 3]> : tensor<2xi64>, edge_padding_low = dense<[4, 5]> : tensor<2xi64>, interior_padding = dense<0> : tensor<2xi64>} : (tensor<12x4xf32>, tensor<f32>) -> tensor<18x12xf32>

    // CHECK: [[ARG1STOREPTR:%.*]] = spv.AccessChain [[ARG1]]
    // CHECK: spv.Store "StorageBuffer" [[ARG1STOREPTR]], [[VALUE]]
    iree.store_output(%2 : tensor<18x12xf32>, %arg1 : memref<18x12xf32>)
    iree.return
  }
}

// -----

module {
  // CHECK: spv.func @pad_no_op
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<48 x f32 [4]> [0]>, StorageBuffer>
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<48 x f32 [4]> [0]>, StorageBuffer>
  func @pad_no_op(%arg0 : memref<12x4xf32>, %arg1 : memref<12x4xf32>)
  attributes  {iree.executable.export, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {

  // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0]]
    // CHECK: [[INPUTVAL:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]] : f32
    %0 = iree.load_input(%arg0 : memref<12x4xf32>) : tensor<12x4xf32>
    %1 = constant dense<0.0> : tensor<f32>

    // CHECK-NOT: spv.Select
    %2 = "xla_hlo.pad"(%0, %1) {edge_padding_high = dense<[0, 0]> : tensor<2xi64>, edge_padding_low = dense<[0, 0]> : tensor<2xi64>, interior_padding = dense<0> : tensor<2xi64>} : (tensor<12x4xf32>, tensor<f32>) -> tensor<12x4xf32>

    // CHECK: [[ARG1STOREPTR:%.*]] = spv.AccessChain [[ARG1]]
    // CHECK: spv.Store "StorageBuffer" [[ARG1STOREPTR]], [[INPUTVAL]]
    iree.store_output(%2 : tensor<12x4xf32>, %arg1 : memref<12x4xf32>)
    iree.return
  }
}

// -----

module {
  // CHECK: spv.func @pad_zero_interior
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<48 x f32 [4]> [0]>, StorageBuffer>
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<522 x f32 [4]> [0]>, StorageBuffer>
  func @pad_zero_interior(%arg0 : memref<12x4xf32>, %arg1 : memref<29x18xf32>)
  attributes  {iree.executable.export, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0]]
    // CHECK: [[INPUTVAL:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]] : f32
    %0 = iree.load_input(%arg0 : memref<12x4xf32>) : tensor<12x4xf32>

    // CHECK: [[PADVAL:%.*]] = spv.constant 0.000000e+00 : f32
    %1 = constant dense<0.0> : tensor<f32>

    // CHECK: [[VALUE:%.*]] = spv.Select {{%.*}}, [[INPUTVAL]], [[PADVAL]] : i1, f32
    %2 = "xla_hlo.pad"(%0, %1) {edge_padding_high = dense<[2, 3]> : tensor<2xi64>, edge_padding_low = dense<[4, 5]> : tensor<2xi64>, interior_padding = dense<[1, 2]> : tensor<2xi64>} : (tensor<12x4xf32>, tensor<f32>) -> tensor<29x18xf32>

    // CHECK: [[ARG1STOREPTR:%.*]] = spv.AccessChain [[ARG1]]
    // CHECK: spv.Store "StorageBuffer" [[ARG1STOREPTR]], [[VALUE]]
    iree.store_output(%2 : tensor<29x18xf32>, %arg1 : memref<29x18xf32>)
    iree.return
  }
}
