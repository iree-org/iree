// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv -verify-diagnostics -o - %s | IreeFileCheck %s

module {
  // CHECK-DAG: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.func @concatenate
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<64 x f32 [4]> [0]>, StorageBuffer>
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<10 x f32 [4]> [0]>, StorageBuffer>
  func @concatenate(%arg0: memref<1x64xf32>, %arg1 : memref<1x10xf32>, %arg2 : memref<1x74xf32>)
  attributes  {iree.executable.export, iree.executable.workgroup_size = [1 : index, 1 : index, 1 : index], iree.ordinal = 0 : i32} {
    // CHECK: spv.loop
    // CHECK: spv.loop
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: spv.loop
    // CHECK: ^{{.*}}([[IV:%.*]]: i32):
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0]]
    // CHECK: [[INPUTVAL0:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]] : f32
    // CHECK: [[ARG1LOADPTR:%.*]] = spv.AccessChain [[ARG1]]
    // CHECK: [[INPUTVAL1:%.*]] = spv.Load "StorageBuffer" [[ARG1LOADPTR]] : f32
    // CHECK: [[TRUE:%.*]] = spv.constant true
    // CHECK: [[SIXTY_FOUR:%.*]] = spv.constant 64 : i32
    // CHECK: [[CHECK:%.*]] = spv.SGreaterThanEqual [[IV]], [[SIXTY_FOUR]] : i32
    // CHECK: [[COND:%.*]] = spv.LogicalAnd [[TRUE]], [[CHECK]] : i1
    // CHECK: [[RESULT:%.*]] = spv.Select [[COND]], [[INPUTVAL1]], [[INPUTVAL0]] : i1, f32
    %0 = iree.load_input(%arg0 : memref<1x64xf32>) : tensor<1x64xf32>
    %1 = iree.load_input(%arg1 : memref<1x10xf32>) : tensor<1x10xf32>
    %2 = "xla_hlo.concatenate"(%0, %1) {dimension = 1 : i64} : (tensor<1x64xf32>, tensor<1x10xf32>) -> tensor<1x74xf32>
    iree.store_output(%2 : tensor<1x74xf32>, %arg2 : memref<1x74xf32>)
    return
  }
}
