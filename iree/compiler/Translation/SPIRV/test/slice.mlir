// RUN: iree-opt -split-input-file -convert-iree-to-spirv -simplify-spirv-affine-exprs=false -verify-diagnostics -o - %s | FileCheck %s

module {
  // VAR1 = (x / 3) + 2
  // VAR3 = (x mod 3) + 1
  // VAR0 = (x / 3)
  // VAR2 = (X mod 3)
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0)
  // CHECK: spv.globalVariable [[ARG1VAR:@.*]] bind(0, 1)
  func @slice_unit_stride(%arg0: memref<6x6xf32>, %arg1: memref<2x3xf32>)
  attributes {iree.executable.export, iree.executable.workload = dense<[6, 1]> : tensor<2xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract [[GLOBALID]][1 : i32]
    // CHECK-LABEL: spv.selection
    // CHECK: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK: [[ZERO:%.*]] = spv.constant 0 : i32
    // CHECK: [[THREE:%.*]] = spv.constant 3 : i32
    // CHECK: [[VAR0:%.*]] = spv.SDiv [[GLOBALIDX]], [[THREE]] : i32
    // CHECK: [[TWO:%.*]] = spv.constant 2 : i32
    // CHECK: [[VAR1:%.*]] = spv.IAdd [[VAR0]], [[TWO]] : i32
    // CHECK: [[VAR2:%.*]] = spv.SMod [[GLOBALIDX]], [[THREE]] : i32
    // CHECK: [[ONE:%.*]] = spv.constant 1 : i32
    // CHECK: [[VAR3:%.*]] = spv.IAdd [[VAR2]], [[ONE]] : i32
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO]], [[VAR1]], [[VAR3]]{{\]}}
    // CHECK: [[VAL0:%.*]]  = spv.Load "StorageBuffer" [[ARG0LOADPTR]] : f32
    // CHECK: [[ARG1PTR:%.*]] = spv._address_of [[ARG1VAR]]
    // CHECK: [[ZERO2:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG1STOREPTR:%.*]] = spv.AccessChain [[ARG1PTR]]{{\[}}[[ZERO2]], [[VAR0]], [[VAR2]]{{\]}}
    // CHECK: spv.Store "StorageBuffer" [[ARG1STOREPTR]], [[VAL0]] : f32
    %0 = iree.load_input(%arg0 : memref<6x6xf32>) : tensor<6x6xf32>
    %1 = "xla_hlo.slice"(%0) {start_indices = dense<[2, 1]> : tensor<2xi64>, limit_indices = dense<[4, 4]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>} : (tensor<6x6xf32>) -> tensor<2x3xf32>
    iree.store_output(%1 : tensor<2x3xf32>, %arg1 : memref<2x3xf32>)
    iree.return
  }
}

// -----

module {
  // VAR1 = (x / 3) + 2
  // VAR4 = (x mod 3) * 2 + 1
  // VAR0 = (x / 3)
  // VAR2 = (x mod 3)
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0)
  // CHECK: spv.globalVariable [[ARG1VAR:@.*]] bind(0, 1)
  func @slice_non_unit_stride(%arg0: memref<6x6xf32>, %arg1: memref<2x3xf32>)
  attributes {iree.executable.export, iree.executable.workload = dense<[6, 1]> : tensor<2xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract %1[0 : i32] : vector<3xi32>
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract %1[1 : i32] : vector<3xi32>
    // CHECK-LABEL: spv.selection
    // CHECK: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK: [[ZERO:%.*]] = spv.constant 0 : i32
    // CHECK: [[THREE:%.*]] = spv.constant 3 : i32
    // CHECK: [[VAR0:%.*]] = spv.SDiv [[GLOBALIDX]], [[THREE]] : i32
    // CHECK: [[TWO:%.*]]  = spv.constant 2 : i32
    // CHECK: [[VAR1:%.*]] = spv.IAdd [[VAR0]], [[TWO]] : i32
    // CHECK: [[VAR2:%.*]]  = spv.SMod [[GLOBALIDX]], [[THREE]] : i32
    // CHECK: [[VAR3:%.*]]  = spv.IMul [[VAR2]], [[TWO]] : i32
    // CHECK: [[ONE:%.*]]  = spv.constant 1 : i32
    // CHECK: [[VAR4:%.*]]  = spv.IAdd [[VAR3]], [[ONE]] : i32
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO]], [[VAR1]], [[VAR4]]{{\]}}
    // CHECK: [[VAL0:%.*]]  = spv.Load "StorageBuffer" [[ARG0LOADPTR]] : f32
    // CHECK: [[ARG1PTR:%.*]] = spv._address_of [[ARG1VAR]]
    // CHECK: [[ZERO2:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG1STOREPTR:%.*]] = spv.AccessChain [[ARG1PTR]]{{\[}}[[ZERO2]], [[VAR0]], [[VAR2]]{{\]}}
    // CHECK: spv.Store "StorageBuffer" [[ARG1STOREPTR]], [[VAL0]] : f32
    %0 = iree.load_input(%arg0 : memref<6x6xf32>) : tensor<6x6xf32>
    %1 = "xla_hlo.slice"(%0) {start_indices = dense<[2, 1]> : tensor<2xi64>, limit_indices = dense<[4, 6]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<6x6xf32>) -> tensor<2x3xf32>
    iree.store_output(%1 : tensor<2x3xf32>, %arg1 : memref<2x3xf32>)
    iree.return
  }
}
