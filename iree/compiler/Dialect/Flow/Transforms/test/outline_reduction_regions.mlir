// RUN: iree-opt -split-input-file -iree-flow-outline-reduction-regions -cse %s | IreeFileCheck %s

func @single_reduction(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
  %cst = constant dense<0.000000e+00> : tensor<f32>
  %cst_0 = constant dense<[4, 1, 1]> : vector<3xi32>
  %0 = flow.reduction.region[%cst_0 : vector<3xi32>](%arg1 = %arg0 : tensor<4x8xf32>, %arg2 = %cst : tensor<f32>) -> tensor<4xf32> {
    %1 = "xla_hlo.reduce"(%arg1, %arg2) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>): // no predecessors
      %2 = xla_hlo.add %arg3, %arg4 : tensor<f32>
      "xla_hlo.return"(%2) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
    flow.return %1 : tensor<4xf32>
  } invocation((%arg1, %arg2) : tensor<f32>) -> tensor<f32> {
    %1 = xla_hlo.add %arg1, %arg2 : tensor<f32>
    flow.return %1 : tensor<f32>
  } {dimensions = dense<1> : vector<1xi32>}
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: flow.executable @single_reduction_reduce_0_dim_0 {
//  CHECK-NEXT:   flow.reduction.entry @single_reduction_reduce_0_dim_0_dispatch apply(@single_reduction_reduce_0_dim_0_invocation) attributes {dimension = 1 : i32}
//  CHECK-NEXT:   module {
//  CHECK-NEXT:     func @single_reduction_reduce_0_dim_0_dispatch(tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
//  CHECK-NEXT:     func @single_reduction_reduce_0_dim_0_invocation(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
//  CHECK-NEXT:       %0 = xla_hlo.add %arg0, %arg1 : tensor<f32>
//  CHECK-NEXT:       return %0 : tensor<f32>
//  CHECK-NEXT:     }
//  CHECK-NEXT:   }
//  CHECK-NEXT: }
//  CHECK-NEXT: func @single_reduction(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
//   CHECK-DAG:   %cst = constant dense<0.000000e+00> : tensor<f32>
//   CHECK-DAG:   %cst_0 = constant dense<[4, 1, 1]> : vector<3xi32>
//  CHECK-NEXT:   %0 = flow.dispatch @single_reduction_reduce_0_dim_0::@single_reduction_reduce_0_dim_0_dispatch[%cst_0 : vector<3xi32>](%arg0, %cst) : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
//  CHECK-NEXT:   return %0 : tensor<4xf32>
//  CHECK-NEXT: }

// -----

func @unrolled_reduction(%arg0: tensor<4x2x8xf32>) -> tensor<4xf32> {
  %cst = constant dense<0.000000e+00> : tensor<f32>
  %cst_0 = constant dense<[4, 1, 1]> : vector<3xi32>
  %0 = flow.reduction.region[%cst_0 : vector<3xi32>](%arg1 = %arg0 : tensor<4x2x8xf32>, %arg2 = %cst : tensor<f32>) -> tensor<4xf32> {
    %1 = "xla_hlo.reduce"(%arg1, %arg2) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>): // no predecessors
      %2 = xla_hlo.add %arg3, %arg4 : tensor<f32>
      "xla_hlo.return"(%2) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<4x2x8xf32>, tensor<f32>) -> tensor<4xf32>
    flow.return %1 : tensor<4xf32>
  } invocation((%arg1, %arg2) : tensor<f32>) -> tensor<f32> {
    %1 = xla_hlo.add %arg1, %arg2 : tensor<f32>
    flow.return %1 : tensor<f32>
  } {dimensions = dense<[1, 2]> : vector<2xi32>}
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: flow.executable @unrolled_reduction_reduce_0_dim_0 {
//  CHECK-NEXT:   flow.reduction.entry @unrolled_reduction_reduce_0_dim_0_dispatch apply(@unrolled_reduction_reduce_0_dim_0_invocation) attributes {dimension = 2 : i32}
//  CHECK-NEXT:   module {
//  CHECK-NEXT:     func @unrolled_reduction_reduce_0_dim_0_dispatch(tensor<4x2x8xf32>, tensor<f32>) -> tensor<4x2xf32>
//  CHECK-NEXT:     func @unrolled_reduction_reduce_0_dim_0_invocation(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
//  CHECK-NEXT:       %0 = xla_hlo.add %arg0, %arg1 : tensor<f32>
//  CHECK-NEXT:       return %0 : tensor<f32>
//  CHECK-NEXT:     }
//  CHECK-NEXT:   }
//  CHECK-NEXT: }
//  CHECK-NEXT: flow.executable @unrolled_reduction_reduce_0_dim_1 {
//  CHECK-NEXT:   flow.reduction.entry @unrolled_reduction_reduce_0_dim_1_dispatch apply(@unrolled_reduction_reduce_0_dim_1_invocation) attributes {dimension = 1 : i32}
//  CHECK-NEXT:   module {
//  CHECK-NEXT:     func @unrolled_reduction_reduce_0_dim_1_dispatch(tensor<4x2xf32>, tensor<f32>) -> tensor<4xf32>
//  CHECK-NEXT:     func @unrolled_reduction_reduce_0_dim_1_invocation(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
//  CHECK-NEXT:       %0 = xla_hlo.add %arg0, %arg1 : tensor<f32>
//  CHECK-NEXT:       return %0 : tensor<f32>
//  CHECK-NEXT:     }
//  CHECK-NEXT:   }
//  CHECK-NEXT: }
//  CHECK-NEXT: func @unrolled_reduction(%arg0: tensor<4x2x8xf32>) -> tensor<4xf32> {
//   CHECK-DAG:   %cst = constant dense<0.000000e+00> : tensor<f32>
//   CHECK-DAG:   %cst_0 = constant dense<[4, 1, 1]> : vector<3xi32>
//  CHECK-NEXT:   %0 = flow.dispatch @unrolled_reduction_reduce_0_dim_0::@unrolled_reduction_reduce_0_dim_0_dispatch[%cst_0 : vector<3xi32>](%arg0, %cst) : (tensor<4x2x8xf32>, tensor<f32>) -> tensor<4x2xf32>
//  CHECK-NEXT:   %1 = flow.dispatch @unrolled_reduction_reduce_0_dim_1::@unrolled_reduction_reduce_0_dim_1_dispatch[%cst_0 : vector<3xi32>](%0, %cst) : (tensor<4x2xf32>, tensor<f32>) -> tensor<4xf32>
//  CHECK-NEXT:   return %1 : tensor<4xf32>
//  CHECK-NEXT: }

// -----

func @multi_reduction(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %cst = constant dense<0.000000e+00> : tensor<f32>
  %cst_0 = constant dense<1.000000e+00> : tensor<f32>
  %cst_1 = constant dense<[4, 1, 1]> : vector<3xi32>
  %0:2 = flow.reduction.region[%cst_1 : vector<3xi32>](%arg2 = %arg0 : tensor<4x8xf32>, %arg3 = %arg1 : tensor<4x8xf32>, %arg4 = %cst : tensor<f32>, %arg5 = %cst_0 : tensor<f32>) -> (tensor<4xf32>, tensor<4xf32>) {
    %1:2 = "xla_hlo.reduce"(%arg2, %arg3, %arg4, %arg5) ( {
    ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>, %arg8: tensor<f32>, %arg9: tensor<f32>): // no predecessors
      %2 = xla_hlo.add %arg6, %arg8 : tensor<f32>
      %3 = xla_hlo.add %arg7, %arg9 : tensor<f32>
      "xla_hlo.return"(%2, %3) : (tensor<f32>, tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<f32>, tensor<f32>) -> (tensor<4xf32>, tensor<4xf32>)
    flow.return %1#0, %1#1 : tensor<4xf32>, tensor<4xf32>
  } invocation((%arg2, %arg3) : tensor<f32>, (%arg4, %arg5) : tensor<f32>) -> (tensor<f32>, tensor<i32>) {
    %1 = xla_hlo.add %arg2, %arg4 : tensor<f32>
    %2 = xla_hlo.add %arg3, %arg5 : tensor<f32>
    flow.return %1, %2 : tensor<f32>, tensor<f32>
  } {dimensions = dense<1> : vector<1xi32>}
  return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: flow.executable @multi_reduction_reduce_0_dim_0 {
//  CHECK-NEXT:   flow.reduction.entry @multi_reduction_reduce_0_dim_0_dispatch apply(@multi_reduction_reduce_0_dim_0_invocation) attributes {dimension = 1 : i32}
//  CHECK-NEXT:   module {
//  CHECK-NEXT:     func @multi_reduction_reduce_0_dim_0_dispatch(tensor<4x8xf32>, tensor<4x8xf32>, tensor<f32>, tensor<f32>) -> (tensor<4xf32>, tensor<4xf32>)
//  CHECK-NEXT:     func @multi_reduction_reduce_0_dim_0_invocation(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
//  CHECK-NEXT:       %0 = xla_hlo.add %arg0, %arg2 : tensor<f32>
//  CHECK-NEXT:       %1 = xla_hlo.add %arg1, %arg3 : tensor<f32>
//  CHECK-NEXT:       return %0, %1 : tensor<f32>, tensor<f32>
//  CHECK-NEXT:     }
//  CHECK-NEXT:   }
//  CHECK-NEXT: }
//  CHECK-NEXT: func @multi_reduction(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
//   CHECK-DAG:   %cst = constant dense<0.000000e+00> : tensor<f32>
//   CHECK-DAG:   %cst_0 = constant dense<1.000000e+00> : tensor<f32>
//   CHECK-DAG:   %cst_1 = constant dense<[4, 1, 1]> : vector<3xi32>
//  CHECK-NEXT:   %0:2 = flow.dispatch @multi_reduction_reduce_0_dim_0::@multi_reduction_reduce_0_dim_0_dispatch[%cst_1 : vector<3xi32>](%arg0, %arg1, %cst, %cst_0) : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<f32>, tensor<f32>) -> (tensor<4xf32>, tensor<4xf32>)
//  CHECK-NEXT:   return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
//  CHECK-NEXT: }
