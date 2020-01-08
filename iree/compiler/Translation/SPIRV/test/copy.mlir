// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv -verify-diagnostics -o - %s | IreeFileCheck %s

module {
  // CHECK:spv.module "Logical" "GLSL450"
  // CHECK-DAG: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: func [[FN:@simple_load_store]]
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]*]]: !spv.ptr<!spv.struct<!spv.array<504 x i32 [4]> [0]>, StorageBuffer>
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]*]]: !spv.ptr<!spv.struct<!spv.array<504 x i32 [4]> [0]>, StorageBuffer>
  // CHECK-SAME: {{spirv|spv}}.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}
  func @simple_load_store(%arg0: memref<12x42xi32>, %arg1: memref<12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract [[GLOBALID]][1 : i32]
    // CHECK: [[COND0:%.*]] = spv.constant true
    // CHECK: [[DIMX:%.*]] = spv.constant 42 : i32
    // CHECK: [[CHECKDIMX:%.*]] = spv.SLessThan [[GLOBALIDX]], [[DIMX]] : i32
    // CHECK: [[COND1:%.*]] = spv.LogicalAnd [[COND0]], [[CHECKDIMX]] : i1
    // CHECK: [[DIMY:%.*]] = spv.constant 12 : i32
    // CHECK: [[CHECKDIMY:%.*]] = spv.SLessThan [[GLOBALIDY]], [[DIMY]] : i32
    // CHECK: [[COND2:%.*]] = spv.LogicalAnd [[COND1]], [[CHECKDIMY]] : i1
    // CHECK: spv.selection {
    // CHECK: spv.BranchConditional [[COND2]], [[BB1:\^.*]], [[BB2:\^.*]]
    // CHECK: [[BB1]]:
    // CHECK: [[ZERO1:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0]]{{\[}}[[ZERO1]], {{%.*}}{{\]}}
    // CHECK: [[VAL:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]]
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    %1 = "xla_hlo.copy"(%0) : (tensor<12x42xi32>) -> tensor<12x42xi32>
    // CHECK: [[ZERO2:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG1STOREPTR:%.*]] = spv.AccessChain [[ARG1]]{{\[}}[[ZERO2]], {{%.*}}{{\]}}
    // CHECK: spv.Store "StorageBuffer" [[ARG1STOREPTR]], [[VAL]]
    // CHECK: spv.Branch [[BB2]]
    iree.store_output(%1 : tensor<12x42xi32>, %arg1 : memref<12x42xi32>)
    // CHECK: spv.Return
    iree.return
  }
}
