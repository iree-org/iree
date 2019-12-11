// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv -verify-diagnostics -o - %s | IreeFileCheck %s

module {
  // CHECK-DAG: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: func @transpose_add
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]*]]: !spv.ptr<!spv.struct<!spv.array<12 x !spv.array<12 x f32 [4]> [48]> [0]>, StorageBuffer>
  func @transpose_add(%arg0: memref<12x12xf32>, %arg1: memref<12x12xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[12, 12, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract [[GLOBALID]][1 : i32]
    // CHECK: spv.selection
    // CHECK: [[ZERO1:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0]]{{\[}}[[ZERO1]], [[GLOBALIDY]], [[GLOBALIDX]]{{\]}}
    // CHECK: [[VAL1:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]]
    // CHECK: [[ZERO2:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG1LOADPTR:%.*]] = spv.AccessChain [[ARG0]]{{\[}}[[ZERO2]], [[GLOBALIDX]], [[GLOBALIDY]]{{\]}}
    // CHECK: [[VAL2:%.*]] = spv.Load "StorageBuffer" [[ARG1LOADPTR]]
    %0 = iree.load_input(%arg0 : memref<12x12xf32>) : tensor<12x12xf32>
    %1 = "xla_hlo.transpose"(%0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<12x12xf32>) -> tensor<12x12xf32>
    // CHECK: [[RESULT:%.*]] = spv.FAdd [[VAL1]], [[VAL2]]
    %2 = "xla_hlo.add"(%0, %1) : (tensor<12x12xf32>, tensor<12x12xf32>) -> tensor<12x12xf32>
    iree.store_output(%2 : tensor<12x12xf32>, %arg1 : memref<12x12xf32>)
    iree.return
  }
}
