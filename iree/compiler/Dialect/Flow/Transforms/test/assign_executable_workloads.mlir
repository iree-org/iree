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

// RUN: iree-opt -split-input-file -iree-flow-assign-executable-workloads %s | FileCheck %s --dump-input=fail

flow.executable @singleStaticWorkload_ex_dispatch_0 {
  // CHECK-LABEL: flow.dispatch.entry @singleStaticWorkload_rgn_dispatch_0
  // CHECK-SAME: workgroup_size = dense<[32, 1, 1]> : vector<3xi32>
  // CHECK-SAME: workload = dense<[4, 1, 1]> : vector<3xi32>
  flow.dispatch.entry @singleStaticWorkload_rgn_dispatch_0
  module {
    func @singleStaticWorkload_rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = addf %arg0, %arg0 : tensor<4xf32>
      %1 = subf %0, %arg0 : tensor<4xf32>
      %2 = mulf %1, %arg0 : tensor<4xf32>
      return %2 : tensor<4xf32>
    }
  }
}
func @singleStaticWorkload(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %cst = constant dense<[4, 1, 1]> : vector<3xi32>
  %0 = flow.dispatch @singleStaticWorkload_ex_dispatch_0::@singleStaticWorkload_rgn_dispatch_0[%cst : vector<3xi32>](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

flow.executable @reduction_ex_reduce_0_dim_0 {
  // CHECK-LABEL: flow.reduction.entry @reduction_rgn_reduce_0_dim_0_entry
  // CHECK-SAME: workgroup_size = dense<[32, 1, 1]> : vector<3xi32>
  // CHECK-SAME: workload = dense<[4, 1, 1]> : vector<3xi32>
  flow.reduction.entry @reduction_rgn_reduce_0_dim_0_entry apply(@reduction_rgn_reduce_0_dim_0) attributes {dimension = 1 : i32}
  module {
    func @reduction_rgn_reduce_0_dim_0_entry(tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
    func @reduction_rgn_reduce_0_dim_0(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
      %0 = xla_hlo.add %arg0, %arg1 : tensor<f32>
      return %0 : tensor<f32>
    }
  }
}
func @reduction(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
  %cst = constant dense<0.000000e+00> : tensor<f32>
  %cst_0 = constant dense<[4, 1, 1]> : vector<3xi32>
  %0 = flow.dispatch @reduction_ex_reduce_0_dim_0::@reduction_rgn_reduce_0_dim_0_entry[%cst_0 : vector<3xi32>](%arg0, %cst) : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
