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

// Tests printing and parsing of dispatch region ops.

// RUN: iree-opt -split-input-file %s | iree-opt | FileCheck %s --dump-input=fail

// CHECK-LABEL: @singleArg
func @singleArg(%arg0 : tensor<?xf32>) {
  // CHECK-NEXT: %0 = "some.shape"
  // CHECK-NEXT: flow.dispatch.region[%0 : vector<3xi32>](%arg1 = %arg0 : tensor<?xf32>) {
  // CHECK-NEXT:   flow.return
  // CHECK-NEXT: }
  %workload = "some.shape"(%arg0) : (tensor<?xf32>) -> vector<3xi32>
  flow.dispatch.region[%workload : vector<3xi32>](%i0 = %arg0 : tensor<?xf32>) {
    flow.return
  }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: @multipleArgs
func @multipleArgs(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>) {
  // CHECK-NEXT: %0 = "some.shape"
  // CHECK-NEXT: flow.dispatch.region[%0 : vector<3xi32>](%arg2 = %arg0 : tensor<?xf32>, %arg3 = %arg1 : tensor<?xf32>) {
  // CHECK-NEXT:   flow.return
  // CHECK-NEXT: }
  %workload = "some.shape"(%arg0) : (tensor<?xf32>) -> vector<3xi32>
  flow.dispatch.region[%workload : vector<3xi32>](%i0 = %arg0 : tensor<?xf32>, %i1 = %arg1 : tensor<?xf32>) {
    flow.return
  }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: @singleResult
func @singleResult(%arg0 : tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-NEXT: %0 = "some.shape"
  // CHECK-NEXT: %1 = flow.dispatch.region[%0 : vector<3xi32>](%arg1 = %arg0 : tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-NEXT:   flow.return %arg1 : tensor<?xf32>
  // CHECK-NEXT: }
  %workload = "some.shape"(%arg0) : (tensor<?xf32>) -> vector<3xi32>
  %ret0 = flow.dispatch.region[%workload : vector<3xi32>](%i0 = %arg0 : tensor<?xf32>) -> tensor<?xf32> {
    flow.return %i0 : tensor<?xf32>
  }
  // CHECK-NEXT: return %1 : tensor<?xf32>
  return %ret0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @multipleResults
func @multipleResults(%arg0 : tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
  // CHECK-NEXT: %0 = "some.shape"
  // CHECK-NEXT: %1:2 = flow.dispatch.region[%0 : vector<3xi32>](%arg1 = %arg0 : tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
  // CHECK-NEXT:   flow.return %arg1, %arg1 : tensor<?xf32>, tensor<?xf32>
  // CHECK-NEXT: }
  %workload = "some.shape"(%arg0) : (tensor<?xf32>) -> vector<3xi32>
  %ret0, %ret1 = flow.dispatch.region[%workload : vector<3xi32>](%i0 = %arg0 : tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
    flow.return %i0, %i0 : tensor<?xf32>, tensor<?xf32>
  }
  // CHECK-NEXT: return %1#0, %1#1 : tensor<?xf32>, tensor<?xf32>
  return %ret0, %ret1 : tensor<?xf32>, tensor<?xf32>
}
