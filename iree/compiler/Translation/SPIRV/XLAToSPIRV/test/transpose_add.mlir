// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv -verify-diagnostics -o - %s | IreeFileCheck %s

module {
  // CHECK: spv.func @transpose_add
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]*]]: !spv.ptr<!spv.struct<!spv.array<144 x f32 [4]> [0]>, StorageBuffer>
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]*]]: !spv.ptr<!spv.struct<!spv.array<144 x f32 [4]> [0]>, StorageBuffer>
  func @transpose_add(%arg0: memref<12x12xf32>, %arg1: memref<12x12xf32>)
  attributes  {iree.executable.export, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0]]
    // CHECK: [[VAL1:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]]
    // CHECK: [[ARG1LOADPTR:%.*]] = spv.AccessChain [[ARG0]]
    // CHECK: [[VAL2:%.*]] = spv.Load "StorageBuffer" [[ARG1LOADPTR]]
    %0 = iree.load_input(%arg0 : memref<12x12xf32>) : tensor<12x12xf32>
    %1 = "xla_hlo.transpose"(%0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<12x12xf32>) -> tensor<12x12xf32>
    // CHECK: [[RESULT:%.*]] = spv.FAdd [[VAL1]], [[VAL2]]
    // CHECK: [[ARG1STOREPTR:%.*]] = spv.AccessChain [[ARG1]]
    // CHECK: spv.Store "StorageBuffer" [[ARG1STOREPTR]], [[RESULT]]
    %2 = "xla_hlo.add"(%0, %1) : (tensor<12x12xf32>, tensor<12x12xf32>) -> tensor<12x12xf32>
    iree.store_output(%2 : tensor<12x12xf32>, %arg1 : memref<12x12xf32>)
    return
  }
}
