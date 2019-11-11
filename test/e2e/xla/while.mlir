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

// A simple while loop example.

// RUN: iree-run-mlir %s --target_backends=interpreter-bytecode --input_values="f32=[1]\nf32=[3]" --noexport_all --noprint_mlir | FileCheck %s --implicit-check-not="[" --implicit-check-not="]" --dump-input=fail

// CHECK-LABEL: EXEC @main
func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> attributes { iree.module.export }  {
  %0 = "xla_hlo.while"(%arg0) ( {
  ^bb0(%arg2: tensor<f32>):
    %1 = "xla_hlo.compare"(%arg2, %arg1) {comparison_direction = "LT", name = "compare.2"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "xla_hlo.return"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg2: tensor<f32>):
    %1 = xla_hlo.add %arg2, %arg2 {name = "compare.0"} : tensor<f32>
    "xla_hlo.return"(%1) : (tensor<f32>) -> ()
  }) : (tensor<f32>) -> tensor<f32>

  return %0 : tensor<f32>
}

// CHECK: f32=4
