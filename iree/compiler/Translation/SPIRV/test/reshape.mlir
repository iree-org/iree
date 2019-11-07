// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv -verify-diagnostics -o - %s | FileCheck %s

module {
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0) : !spv.ptr<!spv.struct<!spv.array<24 x !spv.array<21 x i32 [4]> [84]> [0]>, StorageBuffer>
  func @reshape_2D_2D(%arg0: memref<24x21xi32>, %arg1: memref<12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract [[GLOBALID]][1 : i32]
    // CHECK-LABEL: spv.selection
    // CHECK-DAG: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK-DAG: [[ZERO1:%.*]] = spv.constant 0 : i32
    // CHECK-DAG: [[STRIDE0:%.*]] = spv.constant 42 : i32
    // CHECK-DAG: [[L1:%.*]] = spv.IMul [[GLOBALIDY]], [[STRIDE0]] : i32
    // CHECK-DAG: [[L2:%.*]] = spv.IAdd [[L1]], [[GLOBALIDX]] : i32
    // CHECK-DAG: [[STRIDE2:%.*]] = spv.constant 21 : i32
    // CHECK-DAG: [[INDEX0:%.*]] = spv.SDiv [[L2]], [[STRIDE2]] : i32
    // CHECK-DAG: [[INDEX1:%.*]] = spv.SMod [[L2]], [[STRIDE2]] : i32
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO1]], [[INDEX0]], [[INDEX1]]{{\]}}
    %0 = iree.load_input(%arg0 : memref<24x21xi32>) : tensor<24x21xi32>
    %1 = "xla_hlo.reshape"(%0) : (tensor<24x21xi32>) -> tensor<12x42xi32>
    iree.store_output(%1 : tensor<12x42xi32>, %arg1 : memref<12x42xi32>)
    iree.return
  }
}

// -----

module {
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0) : !spv.ptr<!spv.struct<!spv.array<4 x !spv.array<6 x !spv.array<21 x i32 [4]> [84]> [504]> [0]>, StorageBuffer>
  func @reshape_3D_2D(%arg0: memref<4x6x21xi32>, %arg1: memref<12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK-LABEL: spv.selection
    // CHECK-DAG: [[STRIDE1:%.*]] = spv.constant 126 : i32
    // CHECK-DAG: [[I0:%.*]] = spv.SDiv [[L1:%.*]], [[STRIDE2]] : i32
    // CHECK-DAG: [[I1:%.*]] = spv.SMod [[L1]], [[STRIDE2]] : i32
    // CHECK-DAG: [[STRIDE2:%.*]] = spv.constant 21 : i32
    // CHECK-DAG: [[I2:%.*]] = spv.SDiv [[I1]], [[STRIDE2]] : i32
    // CHECK-DAG: [[I3:%.*]] = spv.SMod [[I1]], [[STRIDE2]] : i32
    // CHECK-DAG: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK-DAG: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO1]], [[I0]], [[I2]], [[I3]]{{\]}}
    %0 = iree.load_input(%arg0 : memref<4x6x21xi32>) : tensor<4x6x21xi32>
    %1 = "xla_hlo.reshape"(%0) : (tensor<4x6x21xi32>) -> tensor<12x42xi32>
    iree.store_output(%1 : tensor<12x42xi32>, %arg1 : memref<12x42xi32>)
    iree.return
  }
}

// -----

module {
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0) : !spv.ptr<!spv.struct<!spv.array<24 x !spv.array<21 x i32 [4]> [84]> [0]>, StorageBuffer>
  // CHECK: spv.globalVariable [[ARG1VAR:@.*]] bind(0, 1) : !spv.ptr<!spv.struct<!spv.array<12 x !spv.array<6 x !spv.array<7 x i32 [4]> [28]> [168]> [0]>, StorageBuffer>
  func @reshape_2D_3D(%arg0: memref<24x21xi32>, %arg1: memref<12x6x7xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract [[GLOBALID]][1 : i32]
    // CHECK-LABEL: spv.selection
    // CHECK: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK-DAG: [[ZERO:%.*]] = spv.constant 0 : i32
    // CHECK-DAG: [[FORTYTWO:%.*]] = spv.constant 42 : i32
    // CHECK-DAG: [[I1:%.*]] = spv.IMul [[GLOBALIDY]], [[FORTYTWO]] : i32
    // CHECK-DAG: [[I2:%.*]] = spv.IAdd [[I1]], [[GLOBALIDX]] : i32
    // CHECK-DAG: [[I3:%.*]] = spv.SDiv [[I2]], [[FORTYTWO]] : i32
    // CHECK-DAG: [[I4:%.*]] = spv.IMul [[I3]], [[FORTYTWO]] : i32
    // CHECK-DAG: [[I5:%.*]] = spv.SMod [[I2]], [[FORTYTWO]] : i32
    // CHECK: [[SEVEN:%.*]] = spv.constant 7 : i32
    // CHECK-DAG: [[I6:%.*]] = spv.SDiv [[I5]], [[SEVEN]] : i32
    // CHECK-DAG: [[I7:%.*]] = spv.IMul [[I6]], [[SEVEN]] : i32
    // CHECK-DAG: [[I8:%.*]] = spv.IAdd [[I4]], [[I7]] : i32
    // CHECK-DAG: [[I9:%.*]] = spv.SMod [[I5]], [[SEVEN]] : i32
    // CHECK-DAG: [[I10:%.*]] = spv.IAdd [[I8]], [[I9]] : i32
    // CHECK: [[TWENTYONE:%.*]] = spv.constant 21 : i32
    // CHECK-DAG: [[I11:%.*]] = spv.SDiv [[I10]], [[TWENTYONE]] : i32
    // CHECK-DAG: [[I12:%.*]] = spv.SMod [[I10]], [[TWENTYONE]] : i32
    // CHECK: {{%.*}} = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO]], [[I11]], [[I12]]
    // CHECK-DAG: [[ARG1PTR:%.*]] = spv._address_of [[ARG1VAR]]
    // CHECK-DAG: [[ZERO2:%.*]] = spv.constant 0 : i32
    // CHECK: {{%.*}} = spv.AccessChain [[ARG1PTR]]{{\[}}[[ZERO2]], [[I3]], [[I6]], [[I9]]{{\]}}
    %0 = iree.load_input(%arg0 : memref<24x21xi32>) : tensor<24x21xi32>
    %1 = "xla_hlo.reshape"(%0) : (tensor<24x21xi32>) -> tensor<12x6x7xi32>
    iree.store_output(%1 : tensor<12x6x7xi32>, %arg1 : memref<12x6x7xi32>)
    iree.return
  }
}
