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

// RUN: iree-run-mlir --target_backends=interpreter-bytecode %s --input_values="2x6xf32= 1 2 3 4 5 6 7 8 9 10 11 12" | IreeFileCheck %s
// RUN: iree-run-mlir --target_backends=vulkan-spirv %s --input_values="2x6xf32= 1 2 3 4 5 6 7 8 9 10 11 12" | IreeFileCheck %s

// CHECK-LABEL: EXEC @reshape_3D_2D
func @reshape_3D_2D(%arg : tensor<2x6xf32>) -> tensor<2x1x6xf32> {
  %result = "xla_hlo.reshape"(%arg) : (tensor<2x6xf32>) -> tensor<2x1x6xf32>
  return %result : tensor<2x1x6xf32>
}
// CHECK: 2x1x6xf32={{\[}}[1 2 3 4 5 6]{{\]}}{{\[}}[7 8 9 10 11 12]{{\]}}
