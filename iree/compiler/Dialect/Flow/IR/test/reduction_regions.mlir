// Tests printing and parsing of reduction region ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @singleReduction
func @singleReduction(%arg0 : tensor<5x1xf32>) {
  // CHECK: [[WORKLOAD:%.+]] = "some.shape"(%arg0) : (tensor<5x1xf32>) -> vector<3xi32>
  %workload = "some.shape"(%arg0) : (tensor<5x1xf32>) -> vector<3xi32>
  // CHECK: [[INITIALF:%.+]] = "some.constant"() : () -> tensor<f32>
  %initialValueF = "some.constant"() : () -> tensor<f32>
  //      CHECK: = flow.reduction.region[
  // CHECK-SAME:     [[WORKLOAD]] : vector<3xi32>
  // CHECK-SAME:   ](
  // CHECK-SAME:     %arg1 = %arg0 : tensor<5x1xf32>, %arg2 = [[INITIALF]] : tensor<f32>) -> tensor<1xf32> {
  // CHECK-NEXT:   "xla_hlo.reduce"(%arg1, %arg2) ( {
  //      CHECK: } invocation((%arg1, %arg2) : tensor<f32>) -> tensor<f32> {
  // CHECK-NEXT:   = xla_hlo.add %arg1, %arg2 : tensor<f32>
  // CHECK-NEXT:   flow.return %3 : tensor<f32>
  // CHECK-NEXT: } {dimensions = dense<1> : vector<1xi32>}
  %0 = flow.reduction.region[%workload : vector<3xi32>](%arg1 = %arg0 : tensor<5x1xf32>, %arg2 = %initialValueF : tensor<f32>) -> tensor<1xf32> {
    %1 = "xla_hlo.reduce"(%arg1, %arg2) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>): // no predecessors
      %2 = xla_hlo.add %arg3, %arg4 : tensor<f32>
      "xla_hlo.return"(%2) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<5x1xf32>, tensor<f32>) -> tensor<1xf32>
    flow.return %1 : tensor<1xf32>
  } invocation((%arg1, %arg2) : tensor<f32>) -> tensor<f32> {
    %1 = xla_hlo.add %arg1, %arg2 : tensor<f32>
    flow.return %1 : tensor<f32>
  } {dimensions = dense<1> : vector<1xi32>}
  return
}

// -----

// CHECK-LABEL: @fusedReduction
func @fusedReduction(%arg0 : tensor<4x8xf32>, %arg1 : tensor<4x8xi32>) {
  // CHECK: [[WORKLOAD:%.+]] = "some.shape"(%arg0) : (tensor<4x8xf32>) -> vector<3xi32>
  %workload = "some.shape"(%arg0) : (tensor<4x8xf32>) -> vector<3xi32>
  // CHECK: [[INITIALF:%.+]] = "some.constant"() : () -> tensor<f32>
  // CHECK: [[INITIALI:%.+]] = "some.constant"() : () -> tensor<i32>
  %initialValueF = "some.constant"() : () -> tensor<f32>
  %initialValueI = "some.constant"() : () -> tensor<i32>
  //      CHECK: = flow.reduction.region[
  // CHECK-SAME:       [[WORKLOAD]] : vector<3xi32>
  // CHECK-SAME:     ](
  // CHECK-SAME:       %arg2 = %arg0 : tensor<4x8xf32>, %arg3 = %arg1 : tensor<4x8xi32>, %arg4 = [[INITIALF]] : tensor<f32>, %arg5 = [[INITIALI]] : tensor<i32>) -> (tensor<4xf32>, tensor<4xi32>) {
  // CHECK-NEXT:   "xla_hlo.reduce"(%arg2, %arg3, %arg4, %arg5) ( {
  //      CHECK: } invocation((%arg2, %arg3) : tensor<f32>, (%arg4, %arg5) : tensor<i32>) -> (tensor<f32>, tensor<i32>) {
  // CHECK-NEXT:   = xla_hlo.add %arg2, %arg3 : tensor<f32>
  // CHECK-NEXT:   = xla_hlo.add %arg4, %arg5 : tensor<i32>
  // CHECK-NEXT:   flow.return %4, %5 : tensor<f32>, tensor<i32>
  // CHECK-NEXT: } {dimensions = dense<1> : vector<1xi32>}
  %0:2 = flow.reduction.region[%workload : vector<3xi32>](%arg2 = %arg0 : tensor<4x8xf32>, %arg3 = %arg1 : tensor<4x8xi32>, %arg4 = %initialValueF : tensor<f32>, %arg5 = %initialValueI : tensor<i32>) -> (tensor<4xf32>, tensor<4xi32>) {
    %1:2 = "xla_hlo.reduce"(%arg2, %arg3, %arg4, %arg5) ( {
    ^bb0(%arg6: tensor<f32>, %arg7: tensor<i32>, %arg8: tensor<f32>, %arg9: tensor<i32>): // no predecessors
      %2 = xla_hlo.add %arg6, %arg8 : tensor<f32>
      %3 = xla_hlo.add %arg7, %arg9 : tensor<i32>
      "xla_hlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<4x8xi32>, tensor<f32>, tensor<i32>) -> (tensor<4xf32>, tensor<4xi32>)
    flow.return %1#0, %1#1 : tensor<4xf32>, tensor<4xi32>
  } invocation((%arg2, %arg3) : tensor<f32>, (%arg4, %arg5) : tensor<i32>) -> (tensor<f32>, tensor<i32>) {
    %1 = xla_hlo.add %arg2, %arg3 : tensor<f32>
    %2 = xla_hlo.add %arg4, %arg5 : tensor<i32>
    flow.return %1, %2 : tensor<f32>, tensor<i32>
  } {dimensions = dense<1> : vector<1xi32>}
  return
}
