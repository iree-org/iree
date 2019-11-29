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

// RUN: iree-run-mlir %s --target_backends=interpreter-bytecode --input_values="f32=42.0" --output_types="i,i,i,i,i,i" | IreeFileCheck %s

// CHECK-LABEL: EXEC @cmpf
func @cmpf(%42 : f32) -> (i1, i1, i1, i1, i1, i1) { // need at least one arg to avoid constant folding
  %cm1 = constant -1.0 : f32
  %oeq = cmpf "oeq", %42, %cm1 : f32
  %une = cmpf "une", %42, %cm1 : f32
  %olt = cmpf "olt", %42, %cm1 : f32
  %ole = cmpf "ole", %42, %cm1 : f32
  %ogt = cmpf "ogt", %42, %cm1 : f32
  %oge = cmpf "oge", %42, %cm1 : f32
  return %oeq, %une, %olt, %ole, %ogt, %oge : i1, i1, i1, i1, i1, i1
}
// CHECK: i8=0
// CHECK-NEXT: i8=1
// CHECK-NEXT: i8=0
// CHECK-NEXT: i8=0
// CHECK-NEXT: i8=1
// CHECK-NEXT: i8=1
