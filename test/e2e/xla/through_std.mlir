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

// RUN: iree-run-mlir %s | IreeFileCheck %s

// CHECK-LABEL: EXEC @xla_through_stdops
func @xla_through_stdops () -> (tensor<f32>, tensor<f32>) {
  %tf32 = constant dense<1.0> : tensor<f32>
  %0 = "xla_hlo.add"(%tf32, %tf32) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %1 = "xla_hlo.mul"(%tf32, %tf32) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0, %1 : tensor<f32>, tensor<f32>
}
// CHECK: f32=2
// CHECK-NEXT: f32=1
