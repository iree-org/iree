// RUN: iree-opt -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv %s | IreeFileCheck %s

module {
  // CHECK-LABEL: spv.func @foo
  // CHECK-SAME: [[ARG0:%.*]]: !spv.ptr<!spv.struct<!spv.array<50 x f32 [4]> [0]>, StorageBuffer>
  // CHECK-SAME: [[ARG1:%.*]]: !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
  func @foo(%arg0: memref<5x1x10xf32>, %arg1: memref<i64>, %arg2: memref<1x10xf32>)
  attributes  {iree.executable.export, iree.executable.workgroup_size = [32 : index, 1 : index, 1 : index], iree.ordinal = 0 : i32} {
    // CHECK: [[ZERO1:%.*]] = spv.constant 0
    // CHECK: [[LOAD_ADDRESS_ARG1:%.*]] = spv.AccessChain [[ARG1]]{{\[}}[[ZERO1]]{{\]}}
    // CHECK: [[INDEXI64:%.*]] = spv.Load {{".*"}} [[LOAD_ADDRESS_ARG1]]
    // CHECK: [[INDEX:%.*]] = spv.SConvert [[INDEXI64]] : i64 to i32
    // CHECK: [[ZERO2:%.*]] = spv.constant 0
    // CHECK: [[STRIDE:%.*]] = spv.constant 10
    // CHECK: [[OUTER:%.*]] = spv.IMul [[INDEX]], [[STRIDE]]
    // CHECK: [[LINEARIZED:%.*]] = spv.IAdd {{%.*}}, [[OUTER]]
    // CHECK: spv.AccessChain [[ARG0]]{{\[}}[[ZERO2]], [[LINEARIZED]]
    %0 = iree.load_input(%arg0 : memref<5x1x10xf32>) : tensor<5x1x10xf32>
    %1 = iree.load_input(%arg1 : memref<i64>) : tensor<i64>
    %2 = "xla_hlo.gather"(%0, %1) {
      dimension_numbers = {
        collapsed_slice_dims = dense<0> : tensor<1xi64>,
        index_vector_dim = 0 : i64,
        offset_dims = dense<[0, 1]> : tensor<2xi64>,
        start_index_map = dense<0> : tensor<1xi64>
      },
      slice_sizes = dense<[1, 1, 10]> : tensor<3xi64>
    } : (tensor<5x1x10xf32>, tensor<i64>) -> tensor<1x10xf32>
    iree.store_output(%2 : tensor<1x10xf32>, %arg2 : memref<1x10xf32>)
    return
  }
}
