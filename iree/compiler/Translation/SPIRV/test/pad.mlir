// RUN: iree-opt -split-input-file -convert-iree-to-spirv -simplify-spirv-affine-exprs=false -verify-diagnostics -o - %s | FileCheck %s --dump-input=fail

module {
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0)
  // CHECK: spv.globalVariable [[ARG1VAR:@.*]] bind(0, 1)
  func @pad_zero_interior(%arg0 : memref<12x4xf32>, %arg1 : memref<18x12xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[12, 18, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract [[GLOBALID]][1 : i32]
    // CHECK: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK: [[ZERO1:%.*]] = spv.constant 0 : i32
    // CHECK: [[LOWER_PAD0:%.*]] = spv.constant -4 : i32
    // CHECK: [[INDEX0:%.*]] = spv.IAdd [[GLOBALIDY]], [[LOWER_PAD0]] : i32
    // CHECK: [[LOWER_PAD1:%.*]] = spv.constant -5 : i32
    // CHECK: [[INDEX1:%.*]] = spv.IAdd [[GLOBALIDX]], [[LOWER_PAD1]] : i32
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO1]], [[INDEX0]], [[INDEX1]]{{\]}}
    // CHECK: [[INPUTVAL:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]] : f32
    %0 = iree.load_input(%arg0 : memref<12x4xf32>) : tensor<12x4xf32>

    // CHECK: [[PADVAL:%.*]] = spv.constant 0.000000e+00 : f32
    %1 = constant dense<0.0> : tensor<f32>

    // CHECK: [[ZERO2:%.*]] = spv.constant 0 : i32
    // CHECK: [[COND0:%.*]] = spv.constant true
    // CHECK: [[LOWER_PAD0_2:%.*]] = spv.constant 4 : i32
    // CHECK: [[STRIDE0:%.*]] = spv.constant 1 : i32
    // CHECK: [[OPERAND_EXTENT0:%.*]] = spv.constant 12 : i32
    // CHECK: [[TEMP1:%.*]] = spv.IMul [[STRIDE0]], [[OPERAND_EXTENT0]] : i32
    // CHECK: [[BOUND0:%.*]] = spv.IAdd [[LOWER_PAD0_2]], [[TEMP1]] : i32
    // CHECK: [[CHECK_UB0:%.*]] = spv.SLessThan [[GLOBALIDY]], [[BOUND0]] : i32
    // CHECK: [[COND1:%.*]] = spv.LogicalAnd [[COND0]], [[CHECK_UB0]] : i1
    // CHECK: [[CHECK_LB0:%.*]] = spv.SGreaterThanEqual [[GLOBALIDY]], [[LOWER_PAD0_2]] : i32
    // CHECK: [[COND2:%.*]] = spv.LogicalAnd [[COND1]], [[CHECK_LB0]] : i1

    // CHECK: [[LOWER_PAD1_2:%.*]] = spv.constant 5 : i32
    // CHECK: [[STRIDE1:%.*]] = spv.constant 1 : i32
    // CHECK: [[OPERAND_EXTENT1:%.*]] = spv.constant 4 : i32
    // CHECK: [[TEMP2:%.*]] = spv.IMul [[STRIDE1]], [[OPERAND_EXTENT1]] : i32
    // CHECK: [[BOUND1:%.*]] = spv.IAdd [[LOWER_PAD1_2]], [[TEMP2]] : i32
    // CHECK: [[CHECK_UB1:%.*]] = spv.SLessThan [[GLOBALIDX]], [[BOUND1]] : i32
    // CHECK: [[COND3:%.*]] = spv.LogicalAnd [[COND2]], [[CHECK_UB1]] : i1
    // CHECK: [[CHECK_LB1:%.*]] = spv.SGreaterThanEqual [[GLOBALIDX]], [[LOWER_PAD1_2]] : i32
    // CHECK: [[COND4:%.*]] = spv.LogicalAnd [[COND3]], [[CHECK_LB1]] : i1

    // CHECK: [[VALUE:%.*]] = spv.Select [[COND4]], [[INPUTVAL]], [[PADVAL]] : i1, f32
    %2 = "xla_hlo.pad"(%0, %1) {edge_padding_high = dense<[2, 3]> : tensor<2xi64>, edge_padding_low = dense<[4, 5]> : tensor<2xi64>, interior_padding = dense<0> : tensor<2xi64>} : (tensor<12x4xf32>, tensor<f32>) -> tensor<18x12xf32>
    iree.store_output(%2 : tensor<18x12xf32>, %arg1 : memref<18x12xf32>)
    iree.return
  }
}

// -----

module {
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0)
  // CHECK: spv.globalVariable [[ARG1VAR:@.*]] bind(0, 1)
  func @pad_no_op(%arg0 : memref<12x4xf32>, %arg1 : memref<12x4xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[4, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract [[GLOBALID]][1 : i32]
    // CHECK: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK: [[ZERO1:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO1]], [[GLOBALIDY]], [[GLOBALIDX]]{{\]}}
    // CHECK: [[INPUTVAL:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]] : f32
    %0 = iree.load_input(%arg0 : memref<12x4xf32>) : tensor<12x4xf32>
    // CHECK: [[PADVAL:%.*]] = spv.constant 0.000000e+00 : f32
    %1 = constant dense<0.0> : tensor<f32>
    // CHECK: [[ZERO2:%.*]] = spv.constant 0 : i32
    // CHECK: [[COND0:%.*]] = spv.constant true
    // CHECK: {{%.*}} = spv.Select [[COND0]], [[INPUTVAL]], [[PADVAL]] : i1, f32
    %2 = "xla_hlo.pad"(%0, %1) {edge_padding_high = dense<[0, 0]> : tensor<2xi64>, edge_padding_low = dense<[0, 0]> : tensor<2xi64>, interior_padding = dense<0> : tensor<2xi64>} : (tensor<12x4xf32>, tensor<f32>) -> tensor<12x4xf32>
    iree.store_output(%2 : tensor<12x4xf32>, %arg1 : memref<12x4xf32>)
    iree.return
  }
}

// -----

module {
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0)
  // CHECK: spv.globalVariable [[ARG1VAR:@.*]] bind(0, 1)
  func @pad_zero_interior(%arg0 : memref<12x4xf32>, %arg1 : memref<29x18xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[18, 29, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract [[GLOBALID]][1 : i32]
    // CHECK: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK: [[ZERO1:%.*]] = spv.constant 0 : i32
    // CHECK: [[LOWER_PAD0:%.*]] = spv.constant -4 : i32
    // CHECK: [[SUB_PAD0:%.*]] = spv.IAdd [[GLOBALIDY]], [[LOWER_PAD0]] : i32
    // CHECK: [[INTERIOR0:%.*]] = spv.constant 2 : i32
    // CHECK: [[INDEX0:%.*]] = spv.SDiv [[SUB_PAD0]], [[INTERIOR0]] : i32
    // CHECK: [[LOWER_PAD1:%.*]] = spv.constant -5 : i32
    // CHECK: [[SUB_PAD1:%.*]] = spv.IAdd [[GLOBALIDX]], [[LOWER_PAD1]] : i32
    // CHECK: [[INTERIOR1:%.*]] = spv.constant 3 : i32
    // CHECK: [[INDEX1:%.*]] = spv.SDiv [[SUB_PAD1]], [[INTERIOR1]] : i32
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO1]], [[INDEX0]], [[INDEX1]]{{\]}}
    // CHECK: [[INPUTVAL:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]] : f32
    %0 = iree.load_input(%arg0 : memref<12x4xf32>) : tensor<12x4xf32>

    // CHECK: [[PADVAL:%.*]] = spv.constant 0.000000e+00 : f32
    %1 = constant dense<0.0> : tensor<f32>

    // CHECK: [[ZERO2:%.*]] = spv.constant 0 : i32
    // CHECK: [[COND0:%.*]] = spv.constant true
    // CHECK: [[LOWER_PAD0_2:%.*]] = spv.constant 4 : i32
    // CHECK: [[STRIDE0:%.*]] = spv.constant 2 : i32
    // CHECK: [[OPERAND_EXTENT0:%.*]] = spv.constant 12 : i32
    // CHECK: [[TEMP1:%.*]] = spv.IMul [[STRIDE0]], [[OPERAND_EXTENT0]] : i32
    // CHECK: [[BOUND0:%.*]] = spv.IAdd [[LOWER_PAD0_2]], [[TEMP1]] : i32
    // CHECK: [[CHECK_UB0:%.*]] = spv.SLessThan [[GLOBALIDY]], [[BOUND0]] : i32
    // CHECK: [[COND1:%.*]] = spv.LogicalAnd [[COND0]], [[CHECK_UB0]] : i1
    // CHECK: [[CHECK_LB0:%.*]] = spv.SGreaterThanEqual [[GLOBALIDY]], [[LOWER_PAD0_2]] : i32
    // CHECK: [[COND2:%.*]] = spv.LogicalAnd [[COND1]], [[CHECK_LB0]] : i1
    // CHECK: [[TEMP1_1:%.*]] = spv.ISub [[GLOBALIDY]], [[LOWER_PAD0_2]] : i32
    // CHECK: [[TEMP1_2:%.*]] = spv.SMod [[TEMP1_1]], [[STRIDE0]] : i32
    // CHECK: [[CHECK_INTERIOR0:%.*]] = spv.IEqual [[TEMP1_2]], [[ZERO2]] : i32
    // CHECK: [[COND3:%.*]] = spv.LogicalAnd [[COND2]], [[CHECK_INTERIOR0]] : i1

    // CHECK: [[LOWER_PAD1_2:%.*]] = spv.constant 5 : i32
    // CHECK: [[STRIDE1:%.*]] = spv.constant 3 : i32
    // CHECK: [[OPERAND_EXTENT1:%.*]] = spv.constant 4 : i32
    // CHECK: [[TEMP2:%.*]] = spv.IMul [[STRIDE1]], [[OPERAND_EXTENT1]] : i32
    // CHECK: [[BOUND1:%.*]] = spv.IAdd [[LOWER_PAD1_2]], [[TEMP2]] : i32
    // CHECK: [[CHECK_UB1:%.*]] = spv.SLessThan [[GLOBALIDX]], [[BOUND1]] : i32
    // CHECK: [[COND4:%.*]] = spv.LogicalAnd [[COND3]], [[CHECK_UB1]] : i1
    // CHECK: [[CHECK_LB1:%.*]] = spv.SGreaterThanEqual [[GLOBALIDX]], [[LOWER_PAD1_2]] : i32
    // CHECK: [[COND5:%.*]] = spv.LogicalAnd [[COND4]], [[CHECK_LB1]] : i1
    // CHECK: [[TEMP2_1:%.*]] = spv.ISub [[GLOBALIDX]], [[LOWER_PAD1_2]] : i32
    // CHECK: [[TEMP2_2:%.*]] = spv.SMod [[TEMP2_1]], [[STRIDE1]] : i32
    // CHECK: [[CHECK_INTERIOR1:%.*]] = spv.IEqual [[TEMP2_2]], [[ZERO2]] : i32
    // CHECK: [[COND6:%.*]] = spv.LogicalAnd [[COND5]], [[CHECK_INTERIOR1]] : i1

    // CHECK: [[VALUE:%.*]] = spv.Select [[COND6]], [[INPUTVAL]], [[PADVAL]] : i1, f32
    %2 = "xla_hlo.pad"(%0, %1) {edge_padding_high = dense<[2, 3]> : tensor<2xi64>, edge_padding_low = dense<[4, 5]> : tensor<2xi64>, interior_padding = dense<[1, 2]> : tensor<2xi64>} : (tensor<12x4xf32>, tensor<f32>) -> tensor<29x18xf32>
    iree.store_output(%2 : tensor<29x18xf32>, %arg1 : memref<29x18xf32>)
    iree.return
  }
}
