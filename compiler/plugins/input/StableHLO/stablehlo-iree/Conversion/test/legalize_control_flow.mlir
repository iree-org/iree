// RUN: iree-opt --iree-stablehlo-legalize-control-flow %s | FileCheck %s

// CHECK-LABEL: func @while(
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1xi64>) -> tensor<1xi64> {
func.func @while(%arg0: tensor<1xi64>) -> tensor<1xi64> {

  // CHECK: %[[VAL_1:.*]] = scf.while (%[[VAL_2:.*]] = %[[VAL_0]]) : (tensor<1xi64>) -> tensor<1xi64> {
  %0 = "stablehlo.while"(%arg0) ({
  ^bb0(%arg1: tensor<1xi64>):

    // CHECK: %[[VAL_3:.*]] = stablehlo.compare LT, %[[VAL_2]], %[[VAL_2]] {name = "compare.2"} : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %[[VAL_3]] : (tensor<1xi1>) -> tensor<i1>
    // CHECK: %[[VAL_4:.*]] = tensor.extract %[[RESHAPE]][] : tensor<i1>
    // CHECK: scf.condition(%[[VAL_4]]) %[[VAL_2]] : tensor<1xi64>
    %1 = "stablehlo.compare"(%arg1, %arg1) {comparison_direction = #stablehlo<comparison_direction LT>, name = "compare.2"} : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2 = "stablehlo.reshape"(%1) : (tensor<1xi1>) -> tensor<i1>
    "stablehlo.return"(%2) : (tensor<i1>) -> ()

  // CHECK: } do {
  // CHECK: ^bb0(%[[VAL_5:.*]]: tensor<1xi64>):
  },  {
  ^bb0(%arg1: tensor<1xi64>):

    // CHECK: %[[VAL_6:.*]] = stablehlo.add %[[VAL_5]], %[[VAL_5]] {name = "compare.0"} : tensor<1xi64>
    // CHECK: scf.yield %[[VAL_6]] : tensor<1xi64>
    %1 = stablehlo.add %arg1, %arg1 {name = "compare.0"} : tensor<1xi64>
    "stablehlo.return"(%1) : (tensor<1xi64>) -> ()
  }) : (tensor<1xi64>) -> tensor<1xi64>

  // CHECK: return %[[VAL_7:.*]] : tensor<1xi64>
  func.return %0 : tensor<1xi64>
}


// CHECK-LABEL: func @while_multi_operands(
// CHECK-SAME:    %[[VAL_0:.*]]: tensor<3xi32>) -> tuple<tensor<i32>, tensor<3xi32>> {
func.func @while_multi_operands(%arg0: tensor<3xi32>) -> tuple<tensor<i32>, tensor<3xi32>> {

  // CHECK-NEXT: %[[VAL_1:.*]] = stablehlo.constant dense<false> : tensor<i1>
  // CHECK-NEXT: %[[VAL_2:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.constant dense<false> : tensor<i1>
  %1 = stablehlo.constant dense<0> : tensor<i32>

  // CHECK: %[[VAL_3:.*]]:2 = scf.while (%[[VAL_4:.*]] = %[[VAL_2]], %[[VAL_5:.*]] = %[[VAL_0]]) : (tensor<i32>, tensor<3xi32>) -> (tensor<i32>, tensor<3xi32>) {
  %2:2 = "stablehlo.while"(%1, %arg0) ({
  ^bb0(%arg1: tensor<i32> , %arg2: tensor<3xi32> ):

    // CHECK-NEXT: %[[VAL_6:.*]] = stablehlo.constant dense<false> : tensor<i1>
    // CHECK-NEXT: %[[VAL_7:.*]] = stablehlo.constant dense<8> : tensor<i32>
    // CHECK: %[[VAL_8:.*]] = stablehlo.compare LT, %[[VAL_4]], %[[VAL_7]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
    // CHECK: %[[VAL_9:.*]] = tensor.extract %[[VAL_8]][] : tensor<i1>
    // CHECK: scf.condition(%[[VAL_9]]) %[[VAL_4]], %[[VAL_5]] : tensor<i32>, tensor<3xi32>
    %4 = stablehlo.constant dense<false> : tensor<i1>
    %5 = stablehlo.constant dense<8> : tensor<i32>
    %6 = "stablehlo.compare"(%arg1, %5) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%6) : (tensor<i1>) -> ()
  },  {

  // CHECK: } do {
  // CHECK: ^bb0(%[[VAL_10:.*]]: tensor<i32>, %[[VAL_11:.*]]: tensor<3xi32>):
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<3xi32>):

    // CHECK-NEXT: %[[VAL_12:.*]] = stablehlo.constant dense<false> : tensor<i1>
    // CHECK-NEXT: %[[VAL_13:.*]] = stablehlo.constant dense<1> : tensor<i32>
    // CHECK: %[[VAL_14:.*]] = stablehlo.add %[[VAL_10]], %[[VAL_13]] : tensor<i32>
    // CHECK: %[[VAL_15:.*]] = stablehlo.convert %[[VAL_10]] : tensor<i32>
    // CHECK: %[[VAL_16:.*]] = stablehlo.broadcast_in_dim %[[VAL_15]], dims = [] : (tensor<i32>) -> tensor<3xi32>
    // CHECK: %[[VAL_17:.*]] = stablehlo.add %[[VAL_11]], %[[VAL_16]] : tensor<3xi32>
    // CHECK: scf.yield %[[VAL_14]], %[[VAL_17]] : tensor<i32>, tensor<3xi32>
    %4 = stablehlo.constant dense<false> : tensor<i1>
    %5 = stablehlo.constant dense<1> : tensor<i32>
    %6 = stablehlo.add %arg1, %5 : tensor<i32>
    %7 = stablehlo.convert %arg1 : (tensor<i32>) -> tensor<i32>
    %8 = "stablehlo.broadcast_in_dim"(%7) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i32>) -> tensor<3xi32>
    %9 = stablehlo.add %arg2, %8 : tensor<3xi32>
    "stablehlo.return"(%6, %9) : (tensor<i32>, tensor<3xi32>) -> ()
  }) : (tensor<i32>, tensor<3xi32>) -> (tensor<i32>, tensor<3xi32>)

  // CHECK: %[[VAL_18:.*]] = stablehlo.tuple %[[VAL_19:.*]]#0, %[[VAL_19]]#1 {xla_shape = "(s32[], s32[3]{0})"} : tuple<tensor<i32>, tensor<3xi32>>
  // CHECK: return %[[VAL_18]] : tuple<tensor<i32>, tensor<3xi32>>
  %3 = "stablehlo.tuple"(%2#0, %2#1) {xla_shape = "(s32[], s32[3]{0})"} : (tensor<i32>, tensor<3xi32>) -> tuple<tensor<i32>, tensor<3xi32>>
  func.return %3 : tuple<tensor<i32>, tensor<3xi32>>
}

// CHECK-LABEL: func @conditional(
// CHECK-SAME:    %[[VAL_0:.*]]: tensor<f32>) -> tensor<f32> {
func.func @conditional(%arg0: tensor<f32>) -> tensor<f32> {

  // CHECK-NEXT: %[[VAL_1:.*]] = arith.constant dense<1.000000e+01> : tensor<f32>
  %cst = arith.constant dense<1.000000e+01> : tensor<f32>

  // CHECK: %[[VAL_2:.*]] = stablehlo.compare LT, %[[VAL_0]], %[[VAL_1]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK: %[[VAL_3:.*]] = tensor.extract %[[VAL_2]][] : tensor<i1>
  %0 = "stablehlo.compare"(%arg0, %cst) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>

  // CHECK: %[[VAL_4:.*]] = scf.if %[[VAL_3]] -> (tensor<f32>) {
  %1 = "stablehlo.if"(%0) ({

    // CHECK: %[[VAL_5:.*]] = stablehlo.log %[[VAL_0]] : tensor<f32>
    // CHECK: scf.yield %[[VAL_5]] : tensor<f32>
    %2 = stablehlo.log %arg0 : (tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%2) : (tensor<f32>) -> ()

  // CHECK: } else {
  },  {

    // CHECK: %[[VAL_6:.*]] = stablehlo.exponential %[[VAL_0]] : tensor<f32>
    // CHECK: scf.yield %[[VAL_6]] : tensor<f32>
    %2 = stablehlo.exponential %arg0 : (tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%2) : (tensor<f32>) -> ()
  }) : (tensor<i1>) -> tensor<f32>

  // CHECK:           return %[[VAL_7:.*]] : tensor<f32>
  func.return %1 : tensor<f32>
}

// Check that we recursively lower nested ifs.
// CHECK-LABEL: func @conditional_nested(
func.func @conditional_nested(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %cst = arith.constant dense<1.000000e+01> : tensor<f32>

  %cmp1 = "stablehlo.compare"(%arg0, %cst) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>

  // CHECK: scf.if
  %if1 = "stablehlo.if"(%cmp1) ({
    %cmp2 = "stablehlo.compare"(%arg1, %cst) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %log = stablehlo.log %arg0 : (tensor<f32>) -> tensor<f32>

    // CHECK: scf.if
    %if2 = "stablehlo.if"(%cmp2) ({
      "stablehlo.return"(%arg1) : (tensor<f32>) -> ()
    },  {
      "stablehlo.return"(%log) : (tensor<f32>) -> ()
    }) : (tensor<i1>) -> tensor<f32>
    "stablehlo.return"(%if2) : (tensor<f32>) -> ()
  },  {
    %exp = stablehlo.exponential %arg0 : (tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%exp) : (tensor<f32>) -> ()
  }) : (tensor<i1>) -> tensor<f32>

  func.return %if1 : tensor<f32>
}

// Test the two branches case as the common. Following tests verify degenerate
// behavior.
// CHECK-LABEL: func @case2(
// CHECK-SAME:    %[[VAL_0:.*]]: tensor<i32>,
// CHECK-SAME:    %[[VAL_1:.*]]: tensor<4xf32>,
// CHECK-SAME:    %[[VAL_2:.*]]: tensor<4xf32>) -> tensor<4xf32> {
func.func @case2(%arg0 : tensor<i32>, %arg1 : tensor<4xf32>, %arg2 : tensor<4xf32>) -> tensor<4xf32> {

  // CHECK-NEXT: %[[VAL_3:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK: %[[VAL_4:.*]] = stablehlo.compare EQ, %[[VAL_0]], %[[VAL_3]], NOTYPE : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK: %[[VAL_5:.*]] = tensor.extract %[[VAL_4]][] : tensor<i1>
  // CHECK: %[[VAL_6:.*]] = scf.if %[[VAL_5]] -> (tensor<4xf32>) {
  %1 = "stablehlo.case"(%arg0) ({
      // CHECK: %[[VAL_7:.*]] = stablehlo.log %[[VAL_1]] : tensor<4xf32>
      // CHECK: scf.yield %[[VAL_7]] : tensor<4xf32>
      %2 = stablehlo.log %arg1 : (tensor<4xf32>) -> tensor<4xf32>
      "stablehlo.return"(%2) : (tensor<4xf32>) -> ()

  // CHECK: } else {
  }, {
      // CHECK: %[[VAL_8:.*]] = stablehlo.exponential %[[VAL_2]] : tensor<4xf32>
      // CHECK: scf.yield %[[VAL_8]] : tensor<4xf32>
      %3 = stablehlo.exponential %arg2 : (tensor<4xf32>) -> tensor<4xf32>
      "stablehlo.return"(%3) : (tensor<4xf32>) -> ()
  }) : (tensor<i32>) -> tensor<4xf32>

  // CHECK: return %[[VAL_9:.*]] : tensor<4xf32>
  func.return %1 : tensor<4xf32>
}


// CHECK-LABEL: func @case3(
// CHECK-SAME:    %[[VAL_0:.*]]: tensor<i32>,
// CHECK-SAME:    %[[VAL_1:[0-9a-zA-Z]*]]: tensor<4xf32>,
// CHECK-SAME:    %[[VAL_2:.*]]: tensor<4xf32>,
// CHECK-SAME:    %[[VAL_3:.*]]: tensor<4xf32>) -> tensor<4xf32> {
func.func @case3(%arg0 : tensor<i32>, %arg1 : tensor<4xf32>, %arg2 : tensor<4xf32>, %arg3 : tensor<4xf32>) -> tensor<4xf32> {

  // CHECK-NEXT: %[[VAL_4:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK: %[[VAL_5:.*]] = stablehlo.compare EQ, %[[VAL_0]], %[[VAL_4]], NOTYPE : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK: %[[VAL_6:.*]] = tensor.extract %[[VAL_5]][] : tensor<i1>
  // CHECK: %[[VAL_7:.*]] = scf.if %[[VAL_6]] -> (tensor<4xf32>) {
  %1 = "stablehlo.case"(%arg0) ({
      // CHECK: %[[VAL_8:.*]] = stablehlo.log %[[VAL_1]] : tensor<4xf32>
      // CHECK: scf.yield %[[VAL_8]] : tensor<4xf32>
      %2 = stablehlo.log %arg1 : (tensor<4xf32>) -> tensor<4xf32>
      "stablehlo.return"(%2) : (tensor<4xf32>) -> ()

  // CHECK: } else {
  // CHECK-NEXT:   %[[VAL_9:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // CHECK:   %[[VAL_10:.*]] = stablehlo.compare EQ, %[[VAL_0]], %[[VAL_9]], NOTYPE : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK:   %[[VAL_11:.*]] = tensor.extract %[[VAL_10]][] : tensor<i1>
  // CHECK:   %[[VAL_12:.*]] = scf.if %[[VAL_11]] -> (tensor<4xf32>) {
  }, {
      // CHECK: %[[VAL_13:.*]] = stablehlo.exponential %[[VAL_2]] : tensor<4xf32>
      // CHECK: scf.yield %[[VAL_13]] : tensor<4xf32>
      %3 = stablehlo.exponential %arg2 : (tensor<4xf32>) -> tensor<4xf32>
      "stablehlo.return"(%3) : (tensor<4xf32>) -> ()

  // CHECK: } else {
  }, {

      // CHECK: %[[VAL_14:.*]] = stablehlo.floor %[[VAL_3]] : tensor<4xf32>
      // CHECK: scf.yield %[[VAL_14]] : tensor<4xf32>
      %3 = stablehlo.floor %arg3 : (tensor<4xf32>) -> tensor<4xf32>
      "stablehlo.return"(%3) : (tensor<4xf32>) -> ()
  }) : (tensor<i32>) -> tensor<4xf32>
  // CHECK:   scf.yield %[[VAL_15:.*]] : tensor<4xf32>

  // CHECK: return %[[VAL_16:.*]] : tensor<4xf32>
  func.return %1 : tensor<4xf32>
}

// Case with only one branch is inlined rather than lowering.
// CHECK-LABEL: func @case0(
// CHECK-SAME:    %[[VAL_0:.*]]: tensor<i32>,
// CHECK-SAME:    %[[VAL_1:.*]]: tensor<4xf32>) -> tensor<4xf32> {
func.func @case0(%arg0 : tensor<i32>, %arg1 : tensor<4xf32>) -> tensor<4xf32> {
  %1 = "stablehlo.case"(%arg0) ({
      // CHECK: %[[VAL_2:.*]] = stablehlo.log %[[VAL_1]] : tensor<4xf32>
      %2 = stablehlo.log %arg1 : (tensor<4xf32>) -> tensor<4xf32>
      "stablehlo.return"(%2) : (tensor<4xf32>) -> ()
  }) : (tensor<i32>) -> tensor<4xf32>
  // CHECK: return %[[VAL_2]] : tensor<4xf32>
  func.return %1 : tensor<4xf32>
}

// Case with only one branch is inlined. Check that we recursively lower.
// CHECK-LABEL: func @case0_nested(
// CHECK-SAME:    %[[VAL_0:.*]]: tensor<i32>,
// CHECK-SAME:    %[[VAL_1:.*]]: tensor<4xf32>) -> tensor<4xf32> {
func.func @case0_nested(%arg0 : tensor<i32>, %arg1 : tensor<4xf32>) -> tensor<4xf32> {
  %1 = "stablehlo.case"(%arg0) ({
    %2 = "stablehlo.case"(%arg0) ({
      // CHECK: %[[VAL_2:.*]] = stablehlo.log %[[VAL_1]] : tensor<4xf32>
      %3 = stablehlo.log %arg1 : (tensor<4xf32>) -> tensor<4xf32>
      "stablehlo.return"(%3) : (tensor<4xf32>) -> ()
    }) : (tensor<i32>) -> tensor<4xf32>
    "stablehlo.return"(%2) : (tensor<4xf32>) -> ()
  }) : (tensor<i32>) -> tensor<4xf32>
  // CHECK: return %[[VAL_2]] : tensor<4xf32>
  func.return %1 : tensor<4xf32>
}

func.func @while_is_for(%lb: tensor<i32>, %ub: tensor<i32>, %step: tensor<i32>,
                        %foo: tensor<4xf32>) -> tensor<4xf32> {
  %0:2 = stablehlo.while(%i = %lb, %arg0 = %foo) : tensor<i32>, tensor<4xf32> cond {
    %1 = stablehlo.compare LT, %i, %ub : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    %1 = stablehlo.add %i, %step : tensor<i32>
    stablehlo.return %1, %arg0 : tensor<i32>, tensor<4xf32>
  }
  func.return %0#1 : tensor<4xf32>
}

// CHECK-LABEL: @while_is_for
// CHECK-SAME: %[[LB:.*]]: tensor<i32>, %[[UB:.*]]: tensor<i32>, %[[STEP:.*]]: tensor<i32>
// CHECK-SAME: %[[FOO:.*]]: tensor<4xf32>
// CHECK-DAG:  %[[LB_EXT:.*]] = tensor.extract %[[LB]]
// CHECK-DAG:  %[[UB_EXT:.*]] = tensor.extract %[[UB]]
// CHECK-DAG:  %[[STEP_EXT:.*]] = tensor.extract %[[STEP]]
// CHECK-NEXT: %[[RET:.*]]:2 = scf.for %[[I:.*]] = %[[LB_EXT]] to %[[UB_EXT]] step %[[STEP_EXT]]
// CHECK-SAME: iter_args(%[[I2:.*]] = %[[LB]], %[[ARG0:.*]] = %[[FOO]])
// CHECK-NEXT: %[[TENSOR_I:.*]] = tensor.from_elements %[[I]]
// CHECK-NEXT: %[[NEXT_I2:.*]] = stablehlo.add %[[TENSOR_I]], %[[STEP]]
// CHECK-NEXT: scf.yield %[[NEXT_I2]], %[[ARG0]]
// CHECK:      return %[[RET]]#1

func.func @while_is_for_and_unsigned(%lb: tensor<ui32>, %ub: tensor<ui32>,
                                     %step: tensor<ui32>, %foo: tensor<4xf32>)
                                     -> tensor<4xf32> {
  %0:2 = stablehlo.while(%i = %lb, %arg0 = %foo) : tensor<ui32>, tensor<4xf32> cond {
    %1 = stablehlo.compare LT, %i, %ub : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    %1 = stablehlo.add %i, %step : tensor<ui32>
    stablehlo.return %1, %arg0 : tensor<ui32>, tensor<4xf32>
  }
  func.return %0#1 : tensor<4xf32>
}

// CHECK-LABEL: @while_is_for_and_unsigned
// CHECK: scf.while
