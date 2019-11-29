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

// RUN: iree-run-mlir %s --target_backends=interpreter-bytecode --input_values="i32=42" --output_types="i,i,i,i,i,i,i,i,i,i" | IreeFileCheck %s

// CHECK-LABEL: EXEC @cmpi
func @cmpi(%42 : i32) -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) { // need at least one arg to avoid constant folding
  %cm1 = constant -1 : i32
  %eq = cmpi "eq", %42, %cm1 : i32
  %ne = cmpi "ne", %42, %cm1 : i32
  %slt = cmpi "slt", %42, %cm1 : i32
  %sle = cmpi "sle", %42, %cm1 : i32
  %sgt = cmpi "sgt", %42, %cm1 : i32
  %sge = cmpi "sge", %42, %cm1 : i32
  %ult = cmpi "ult", %42, %cm1 : i32
  %ule = cmpi "ule", %42, %cm1 : i32
  %ugt = cmpi "ugt", %42, %cm1 : i32
  %uge = cmpi "uge", %42, %cm1 : i32
  return %eq, %ne, %slt, %sle, %sgt, %sge, %ult, %ule, %ugt, %uge : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}
// CHECK: i8=0
// CHECK-NEXT: i8=1
// CHECK-NEXT: i8=0
// CHECK-NEXT: i8=0
// CHECK-NEXT: i8=1
// CHECK-NEXT: i8=1
// CHECK-NEXT: i8=1
// CHECK-NEXT: i8=1
// CHECK-NEXT: i8=0
// CHECK-NEXT: i8=0
