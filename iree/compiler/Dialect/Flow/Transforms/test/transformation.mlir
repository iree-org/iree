// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: iree-opt -split-input-file -iree-flow-transformation-pipeline %s | IreeFileCheck %s

// CHECK-LABEL: @empty
func @empty() {
  // CHECK-NEXT: return
  return
}

// -----

func @simpleMath(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = xla_hlo.add %arg0, %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: flow.executable @simpleMath_ex_dispatch_0 {
// CHECK-NEXT:   flow.dispatch.entry @simpleMath_rgn_dispatch_0 attributes {
// CHECK-SAME:     workgroup_size = dense<[32, 1, 1]> : vector<3xi32>,
// CHECK-SAME:     workload = dense<[4, 1, 1]> : vector<3xi32>
// CHECK-SAME:   }
// CHECK-NEXT:   module {
// CHECK-NEXT:     func @simpleMath_rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:       %0 = xla_hlo.add %arg0, %arg0 : tensor<4xf32>
// CHECK-NEXT:       return %0 : tensor<4xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: func @simpleMath(%arg0: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:   %cst = constant dense<[4, 1, 1]> : vector<3xi32>
// CHECK-NEXT:   %0 = flow.ex.stream.fragment(%arg1 = %cst : vector<3xi32>, %arg2 = %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:     %1 = flow.dispatch @simpleMath_ex_dispatch_0::@simpleMath_rgn_dispatch_0[%arg1 : vector<3xi32>](%arg2) : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:     flow.return %1 : tensor<4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0 : tensor<4xf32>
// CHECK-NEXT: }

// -----

func @stdElementwiseOps(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = addf %arg0, %arg0 : tensor<4xf32>
  %1 = subf %0, %arg0 : tensor<4xf32>
  %2 = mulf %1, %arg0 : tensor<4xf32>
  return %2 : tensor<4xf32>
}

// CHECK-LABEL: flow.executable @stdElementwiseOps_ex_dispatch_0 {
// CHECK-NEXT:   flow.dispatch.entry @stdElementwiseOps_rgn_dispatch_0 attributes {
// CHECK-SAME:     workgroup_size = dense<[32, 1, 1]> : vector<3xi32>,
// CHECK-SAME:     workload = dense<[4, 1, 1]> : vector<3xi32>
// CHECK-SAME:   }
// CHECK-NEXT:   module {
// CHECK-NEXT:     func @stdElementwiseOps_rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:       %0 = addf %arg0, %arg0 : tensor<4xf32>
// CHECK-NEXT:       %1 = subf %0, %arg0 : tensor<4xf32>
// CHECK-NEXT:       %2 = mulf %1, %arg0 : tensor<4xf32>
// CHECK-NEXT:       return %2 : tensor<4xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: func @stdElementwiseOps(%arg0: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:   %cst = constant dense<[4, 1, 1]> : vector<3xi32>
// CHECK-NEXT:   %0 = flow.ex.stream.fragment(%arg1 = %cst : vector<3xi32>, %arg2 = %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:     %1 = flow.dispatch @stdElementwiseOps_ex_dispatch_0::@stdElementwiseOps_rgn_dispatch_0[%arg1 : vector<3xi32>](%arg2) : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:     flow.return %1 : tensor<4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0 : tensor<4xf32>
// CHECK-NEXT: }

// -----

func @hloElementwiseOps(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = xla_hlo.add %arg0, %arg0 : tensor<4xf32>
  %1 = xla_hlo.sub %0, %arg0 : tensor<4xf32>
  %2 = xla_hlo.mul %1, %arg0 : tensor<4xf32>
  return %2 : tensor<4xf32>
}

// CHECK-LABEL: flow.executable @hloElementwiseOps_ex_dispatch_0 {
// CHECK-NEXT:   flow.dispatch.entry @hloElementwiseOps_rgn_dispatch_0 attributes {
// CHECK-SAME:     workgroup_size = dense<[32, 1, 1]> : vector<3xi32>,
// CHECK-SAME:     workload = dense<[4, 1, 1]> : vector<3xi32>
// CHECK-SAME:   }
// CHECK-NEXT:   module {
// CHECK-NEXT:     func @hloElementwiseOps_rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:       %0 = xla_hlo.add %arg0, %arg0 : tensor<4xf32>
// CHECK-NEXT:       %1 = xla_hlo.sub %0, %arg0 : tensor<4xf32>
// CHECK-NEXT:       %2 = xla_hlo.mul %1, %arg0 : tensor<4xf32>
// CHECK-NEXT:       return %2 : tensor<4xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: func @hloElementwiseOps(%arg0: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:   %cst = constant dense<[4, 1, 1]> : vector<3xi32>
// CHECK-NEXT:   %0 = flow.ex.stream.fragment(%arg1 = %cst : vector<3xi32>, %arg2 = %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:     %1 = flow.dispatch @hloElementwiseOps_ex_dispatch_0::@hloElementwiseOps_rgn_dispatch_0[%arg1 : vector<3xi32>](%arg2) : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:     flow.return %1 : tensor<4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0 : tensor<4xf32>
// CHECK-NEXT: }

// -----

func @interleavedDot(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = xla_hlo.add %arg0, %arg0 : tensor<4x4xf32>
  %1 = "xla_hlo.dot"(%0, %arg0) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = xla_hlo.mul %1, %arg0 : tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}

// CHECK-LABEL: flow.executable @interleavedDot_ex_dispatch_0 {
// CHECK-NEXT:   flow.dispatch.entry @interleavedDot_rgn_dispatch_0 attributes {
// CHECK-SAME:     workgroup_size = dense<[32, 1, 1]> : vector<3xi32>,
// CHECK-SAME:     workload = dense<[4, 4, 1]> : vector<3xi32>
// CHECK-SAME:   }
// CHECK-NEXT:   module {
// CHECK-NEXT:     func @interleavedDot_rgn_dispatch_0(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:       %0 = xla_hlo.add %arg0, %arg0 : tensor<4x4xf32>
// CHECK-NEXT:       return %0 : tensor<4x4xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: flow.executable @interleavedDot_ex_dispatch_1 {
// CHECK-NEXT:   flow.dispatch.entry @interleavedDot_rgn_dispatch_1 attributes {
// CHECK-SAME:     workgroup_size = dense<[32, 1, 1]> : vector<3xi32>,
// CHECK-SAME:     workload = dense<[4, 4, 1]> : vector<3xi32>
// CHECK-SAME:   }
// CHECK-NEXT:   module {
// CHECK-NEXT:     func @interleavedDot_rgn_dispatch_1(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:       %0 = "xla_hlo.dot"(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:       return %0 : tensor<4x4xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: flow.executable @interleavedDot_ex_dispatch_2 {
// CHECK-NEXT:   flow.dispatch.entry @interleavedDot_rgn_dispatch_2 attributes {
// CHECK-SAME:     workgroup_size = dense<[32, 1, 1]> : vector<3xi32>,
// CHECK-SAME:     workload = dense<[4, 4, 1]> : vector<3xi32>
// CHECK-SAME:   }
// CHECK-NEXT:   module {
// CHECK-NEXT:     func @interleavedDot_rgn_dispatch_2(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:       %0 = xla_hlo.mul %arg0, %arg1 : tensor<4x4xf32>
// CHECK-NEXT:       return %0 : tensor<4x4xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: func @interleavedDot(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:   %cst = constant dense<[4, 4, 1]> : vector<3xi32>
// CHECK-NEXT:   %0 = flow.ex.stream.fragment(%arg1 = %cst : vector<3xi32>, %arg2 = %arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:     %1 = flow.dispatch @interleavedDot_ex_dispatch_0::@interleavedDot_rgn_dispatch_0[%arg1 : vector<3xi32>](%arg2) : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:     %2 = flow.dispatch @interleavedDot_ex_dispatch_1::@interleavedDot_rgn_dispatch_1[%arg1 : vector<3xi32>](%1, %arg2) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:     %3 = flow.dispatch @interleavedDot_ex_dispatch_2::@interleavedDot_rgn_dispatch_2[%arg1 : vector<3xi32>](%2, %arg2) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:     flow.return %3 : tensor<4x4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0 : tensor<4x4xf32>
// CHECK-NEXT: }

// -----

func @reduction(%arg0 : tensor<4x8xf32>) -> tensor<4xf32> {
  %0 = constant dense<0.0> : tensor<f32>
  %1 = "xla_hlo.reduce"(%arg0, %0) ( {
  ^bb0(%arg1 : tensor<f32>, %arg2 : tensor<f32>):
    %2 = xla_hlo.add %arg1, %arg2 : tensor<f32>
    "xla_hlo.return"(%2) : (tensor<f32>) -> ()
  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// CHECK-LABEL: flow.executable @reduction_ex_reduce_0_dim_0 {
// CHECK-NEXT:   flow.reduction.entry @reduction_rgn_reduce_0_dim_0_entry apply(@reduction_rgn_reduce_0_dim_0) attributes {
// CHECK-SAME:     dimension = 1 : i32,
// CHECK-SAME:     workgroup_size = dense<[32, 1, 1]> : vector<3xi32>,
// CHECK-SAME:     workload = dense<[4, 1, 1]> : vector<3xi32>
// CHECK-SAME:   }
// CHECK-NEXT:   module {
// CHECK-NEXT:     func @reduction_rgn_reduce_0_dim_0_entry(tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
// CHECK-NEXT:     func @reduction_rgn_reduce_0_dim_0(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
// CHECK-NEXT:       %0 = xla_hlo.add %arg0, %arg1 : tensor<f32>
// CHECK-NEXT:       return %0 : tensor<f32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: func @reduction(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
// CHECK-NEXT:   %cst = constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:   %cst_0 = constant dense<[4, 1, 1]> : vector<3xi32>
// CHECK-NEXT:   %0 = flow.ex.stream.fragment(%arg1 = %cst_0 : vector<3xi32>, %arg2 = %arg0 : tensor<4x8xf32>, %arg3 = %cst : tensor<f32>) -> tensor<4xf32> {
// CHECK-NEXT:     %1 = flow.dispatch @reduction_ex_reduce_0_dim_0::@reduction_rgn_reduce_0_dim_0_entry[%arg1 : vector<3xi32>](%arg2, %arg3) : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
// CHECK-NEXT:     flow.return %1 : tensor<4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0 : tensor<4xf32>
// CHECK-NEXT: }

// -----

func @dynamicUpdateSlice(%operand : tensor<2x4xi32>, %update : tensor<1x1xi32>, %indices_0 : tensor<i32>, %indices_1 : tensor<i32>) -> tensor<2x4xi32> {
  %0 = "xla_hlo.dynamic-update-slice"(%operand, %update, %indices_0, %indices_1) : (tensor<2x4xi32>, tensor<1x1xi32>, tensor<i32>, tensor<i32>) -> tensor<2x4xi32>
  %1 = xla_hlo.add %operand, %0 : tensor<2x4xi32>
  return %1 : tensor<2x4xi32>
}

// CHECK-LABEL: flow.executable @dynamicUpdateSlice_ex_dispatch_0 {
// CHECK-NEXT: flow.dispatch.entry @dynamicUpdateSlice_rgn_dispatch_0 attributes {workgroup_size = dense<[32, 1, 1]> : vector<3xi32>, workload = dense<[4, 2, 1]> : vector<3xi32>}
// CHECK-NEXT:   module {
// CHECK-NEXT:     func @dynamicUpdateSlice_rgn_dispatch_0(%arg0: tensor<2x4xi32>, %arg1: tensor<2x4xi32>) -> tensor<2x4xi32> {
// CHECK-NEXT:       %0 = xla_hlo.add %arg0, %arg1 : tensor<2x4xi32>
// CHECK-NEXT:       return %0 : tensor<2x4xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: func @dynamicUpdateSlice(%arg0: tensor<2x4xi32>, %arg1: tensor<1x1xi32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<2x4xi32> {
// CHECK-NEXT:   %cst = constant dense<[4, 2, 1]> : vector<3xi32>
// CHECK-NEXT:   %0 = flow.tensor.load %arg2 : tensor<i32>
// CHECK-NEXT:   %1 = flow.tensor.load %arg3 : tensor<i32>
// CHECK-NEXT:   %2 = flow.ex.stream.fragment(%arg4 = %arg1 : tensor<1x1xi32>, %arg5 = %arg0 : tensor<2x4xi32>, %arg6 = %0 : i32, %arg7 = %1 : i32, %arg8 = %cst : vector<3xi32>) -> tensor<2x4xi32> {
// CHECK-NEXT:     %3 = flow.tensor.update %arg4, %arg5[%arg6, %arg7] : tensor<1x1xi32> -> tensor<2x4xi32>
// CHECK-NEXT:     %4 = flow.dispatch @dynamicUpdateSlice_ex_dispatch_0::@dynamicUpdateSlice_rgn_dispatch_0[%arg8 : vector<3xi32>](%arg5, %3) : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi32>
// CHECK-NEXT:     flow.return %4 : tensor<2x4xi32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %2 : tensor<2x4xi32>
// CHECK-NEXT: }
