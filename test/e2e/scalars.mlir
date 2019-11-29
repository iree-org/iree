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

// RUN: iree-run-mlir %s | FileCheck %s --enable-var-scope --dump-input=fail

// CHECK-LABEL: EXEC @scalars
func @scalars() -> tensor<f32> {
  %0 = constant dense<2.0> : tensor<f32>
  return %0 : tensor<f32>
}
// CHECK: f32=2
