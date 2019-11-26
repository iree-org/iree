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

// RUN: iree-run-mlir --target_backends=interpreter-bytecode %s --input_values=1xi32=42 --output_types=i | FileCheck %s --enable-var-scope --dump-input=fail

// CHECK-LABEL: EXEC @narrow_int
func @narrow_int(%arg : tensor<1xi32>) -> tensor<1xi8> {
  %0 = "xla_hlo.convert"(%arg) : (tensor<1xi32>) -> tensor<1xi8>
  return %0 : tensor<1xi8>
}
// CHECK: 1xi8=42

// CHECK-LABEL: EXEC @widen_int
func @widen_int(%arg : tensor<1xi32>) -> tensor<1xi64> {
  %0 = "xla_hlo.convert"(%arg) : (tensor<1xi32>) -> tensor<1xi64>
  return %0 : tensor<1xi64>
}
// CHECK: 1xi64=42

