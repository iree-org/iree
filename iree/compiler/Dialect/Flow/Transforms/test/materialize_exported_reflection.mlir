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

// RUN: iree-opt -split-input-file -verify-diagnostics -iree-flow-materialize-exported-reflection %s | IreeFileCheck %s

// -----
// CHECK-LABEL: func @notExported
// CHECK-NOT: iree.reflection
func @notExported(%arg0 : tensor<4x4xi64>) -> tensor<4x4xi64> {
  return %arg0 : tensor<4x4xi64>
}

// -----
// CHECK-LABEL: func @exportedTensor
// CHECK-SAME: iree.reflection = {f_partial = "I10!B7!t7d4d4"}
// CHECK-SAME: iree.reflection = {f_partial = "I10!B7!t7d5d5"}
// CHECK-SAME: iree.reflection = {f_partial = "R10!B7!t7d5d5"}
func @exportedTensor(%arg0 : tensor<4x4xi64>, %arg1 : tensor<5x5xi64>) -> tensor<5x5xi64>
    attributes {iree.module.export}
{
  return %arg1 : tensor<5x5xi64>
}

// -----
// CHECK-LABEL: func @unrecognizedArgument
// CHECK-SAME: iree.reflection = {f_partial = "I4!U1!"}
// expected-warning @+1 {{Argument #0 of function unrecognizedArgument is not a recognized public ABI type and the function may not be invokable by standard tools}}
func @unrecognizedArgument(%arg0 : i1) -> ()
    attributes {iree.module.export}
{
  return
}

// -----
// CHECK-LABEL: func @unrecognizedResult
// CHECK-SAME: iree.reflection = {f_partial = "R4!U1!"}
// expected-warning @+1 {{Result #0 of function unrecognizedResult is not a recognized public ABI type and the function may not be invokable by standard tools}}
func @unrecognizedResult() -> (i1)
    attributes {iree.module.export}
{
  %0 = constant 0 : i1
  return %0 : i1
}

// -----
// CHECK-LABEL: func @dynamicDim
// CHECK-SAME: iree.reflection = {f_partial = "I11!B8!t7d-1d4"}
func @dynamicDim(%arg0 : tensor<?x4xi64>) -> () attributes {iree.module.export}
{
  return
}

// -----
// CHECK-LABEL: func @scalari32
// CHECK-SAME: iree.reflection = {f_partial = "I6!S3!t6"}
func @scalari32(%arg0 : i32) -> () attributes {iree.module.export}
{
  return
}

// -----
// CHECK-LABEL: func @tensorFloat32
// CHECK-SAME: iree.reflection = {f_partial = "I6!B3!d1"}
func @tensorFloat32(%arg0 : tensor<1xf32>) -> () attributes {iree.module.export}
{
  return
}

// -----
// CHECK-LABEL: func @tensorFloat64
// CHECK-SAME: iree.reflection = {f_partial = "I8!B5!t2d1"}
func @tensorFloat64(%arg0 : tensor<1xf64>) -> () attributes {iree.module.export}
{
  return
}

// -----
// CHECK-LABEL: func @tensorFloat16
// CHECK-SAME: iree.reflection = {f_partial = "I8!B5!t1d1"}
func @tensorFloat16(%arg0 : tensor<1xf16>) -> () attributes {iree.module.export}
{
  return
}

// -----
// CHECK-LABEL: func @tensorBfloat16
// CHECK-SAME: iree.reflection = {f_partial = "I8!B5!t3d1"}
func @tensorBfloat16(%arg0 : tensor<1xbf16>) -> () attributes {iree.module.export}
{
  return
}

// -----
// CHECK-LABEL: func @tensorSint8
// CHECK-SAME: iree.reflection = {f_partial = "I8!B5!t4d1"}
func @tensorSint8(%arg0 : tensor<1xi8>) -> () attributes {iree.module.export}
{
  return
}

// -----
// CHECK-LABEL: func @tensorSint16
// CHECK-SAME: iree.reflection = {f_partial = "I8!B5!t5d1"}
func @tensorSint16(%arg0 : tensor<1xi16>) -> () attributes {iree.module.export}
{
  return
}

// -----
// CHECK-LABEL: func @tensorSint32
// CHECK-SAME: iree.reflection = {f_partial = "I8!B5!t6d1"}
func @tensorSint32(%arg0 : tensor<1xi32>) -> () attributes {iree.module.export}
{
  return
}

// -----
// CHECK-LABEL: func @tensorSint64
// CHECK-SAME: iree.reflection = {f_partial = "I8!B5!t7d1"}
func @tensorSint64(%arg0 : tensor<1xi64>) -> () attributes {iree.module.export}
{
  return
}

