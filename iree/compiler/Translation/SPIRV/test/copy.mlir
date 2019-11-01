// RUN: iree-opt -split-input-file -convert-iree-to-spirv -simplify-spirv-affine-exprs=false -verify-diagnostics -o - %s | FileCheck %s

module {
  // CHECK:spv.module "Logical" "GLSL450"
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0) : !spv.ptr<!spv.struct<!spv.array<12 x !spv.array<42 x i32 [4]> [168]> [0]>, StorageBuffer>
  // CHECK: spv.globalVariable [[ARG1VAR:@.*]] bind(0, 1) : !spv.ptr<!spv.struct<!spv.array<12 x !spv.array<42 x i32 [4]> [168]> [0]>, StorageBuffer>
  // CHECK: func [[FN:@simple_load_store]]()
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
    // CHECK: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK: [[ZERO1:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO1]], [[GLOBALIDY]], [[GLOBALIDX]]{{\]}}
    // CHECK: [[VAL:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]]
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    %1 = "xla_hlo.copy"(%0) : (tensor<12x42xi32>) -> tensor<12x42xi32>
    // CHECK: [[ARG1PTR:%.*]] = spv._address_of [[ARG1VAR]]
    // CHECK: [[ZERO2:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG1STOREPTR:%.*]] = spv.AccessChain [[ARG1PTR]]{{\[}}[[ZERO2]], [[GLOBALIDY]], [[GLOBALIDX]]{{\]}}
    // CHECK: spv.Store "StorageBuffer" [[ARG1STOREPTR]], [[VAL]]
    // CHECK: spv.Branch [[BB2]]
    iree.store_output(%1 : tensor<12x42xi32>, %arg1 : memref<12x42xi32>)
    // CHECK: spv.Return
    iree.return
  }
  // CHECK: spv.EntryPoint "GLCompute" [[FN]], [[GLOBALIDVAR]]
  // CHECK: spv.ExecutionMode [[FN]] "LocalSize", 32, 1, 1
}

// -----

module {
  func @simple_load_store_launch_err(%arg0: memref<12x42xi32>, %arg1: memref<12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 1, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    // expected-error @+1 {{unable to map from launch id to element to compute within a workitem}}
    iree.store_output(%0 : tensor<12x42xi32>, %arg1 : memref<12x42xi32>)
    iree.return
  }
}

// -----

module {
  func @simple_load_store_launch_err2(%arg0: memref<12x42xi32>, %arg1: memref<12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42]> : tensor<1xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    // expected-error @+1 {{unable to map from launch id to element to compute within a workitem}}
    iree.store_output(%0 : tensor<12x42xi32>, %arg1 : memref<12x42xi32>)
    iree.return
  }
}

// -----

module {
  func @simple_load_store_launch_err3(%arg0: memref<42xi32>, %arg1: memref<42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<42xi32>) : tensor<42xi32>
    // expected-error @+1 {{unable to map from launch id to element to compute within a workitem}}
    iree.store_output(%0 : tensor<42xi32>, %arg1 : memref<42xi32>)
    iree.return
  }
}
