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

// RUN: iree-run-mlir %s --input_values="5x3xf32=15,14,13,12,11,10,9,8,7,6,5,4,3,2,1\n3x5xf32=15,14,13,12,11,10,9,8,7,6,5,4,3,2,1" | IreeFileCheck %s

// CHECK-LABEL: EXEC @main
func @main(%arg0: tensor<5x3xf32>, %arg1: tensor<3x5xf32>) -> tensor<5x5xf32>
  attributes {iree.module.export} {
  %0 = "xla_hlo.dot"(%arg0, %arg1) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<5x3xf32>, tensor<3x5xf32>) -> tensor<5x5xf32>
  return %0 : tensor<5x5xf32>
}

// CHECK: 5x5xf32=[430 388 346 304 262][340 307 274 241 208][250 226 202 178 154][160 145 130 115 100][70 64 58 52 46]
