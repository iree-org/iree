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

// RUN: iree-run-mlir --target_backends=interpreter-bytecode %s --input_values="i8=4" --output_types=i | FileCheck %s

// CHECK-LABEL: @extract_element
func @extract_element(%arg0: tensor<i8>) -> i8 {
  %cst = constant dense<1> : tensor<i8>
  %0 = addi %cst, %arg0 : tensor<i8>
  %1 = extract_element %0[] : tensor<i8>
  return %1 : i8
}
// CHECK-NEXT: i8=5
