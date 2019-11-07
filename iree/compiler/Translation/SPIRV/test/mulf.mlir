// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv -verify-diagnostics -o - %s | FileCheck %s

module {
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0)
  // CHECK: spv.globalVariable [[ARG1VAR:@.*]] bind(0, 1)
  // CHECK: spv.globalVariable [[ARG2VAR:@.*]] bind(0, 2)
  func @mul_1D(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[4, 1, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK-LABEL: spv.selection
    // CHECK: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK: [[ZERO1:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO1]], [[GLOBALIDX]]{{\]}}
    // CHECK: [[VAL1:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]]
    %0 = iree.load_input(%arg0 : memref<4xf32>) : tensor<4xf32>
    // CHECK: [[ARG1PTR:%.*]] = spv._address_of [[ARG1VAR]]
    // CHECK: [[ZERO2:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG1LOADPTR:%.*]] = spv.AccessChain [[ARG1PTR]]{{\[}}[[ZERO2]], [[GLOBALIDX]]{{\]}}
    // CHECK: [[VAL2:%.*]] = spv.Load "StorageBuffer" [[ARG1LOADPTR]]
    %1 = iree.load_input(%arg1 : memref<4xf32>) : tensor<4xf32>
    // CHECK: [[RESULT:%.*]] = spv.FMul [[VAL1]], [[VAL2]]
    %2 = mulf %0, %1 : tensor<4xf32>
    // CHECK: [[ARG2PTR:%.*]] = spv._address_of [[ARG2VAR]]
    // CHECK: [[ZERO3:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG2STOREPTR:%.*]] = spv.AccessChain [[ARG2PTR]]{{\[}}[[ZERO3]], [[GLOBALIDX]]{{\]}}
    // CHECK: spv.Store "StorageBuffer" [[ARG2STOREPTR]], [[RESULT]]
    iree.store_output(%2 : tensor<4xf32>, %arg2 : memref<4xf32>)
    iree.return
  }
}
