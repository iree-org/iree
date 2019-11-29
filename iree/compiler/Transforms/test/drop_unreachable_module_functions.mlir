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

// RUN: iree-opt %s -iree-drop-unreachable-module-functions -split-input-file | IreeFileCheck %s --implicit-check-not @unused

// CHECK-LABEL: @exportedModuleFn
func @exportedModuleFn(%arg0 : memref<?xf32>) -> memref<?xf32>
    attributes {iree.module.export} {
  // CHECK: iree_hl_seq.call @fn1
  %0 = iree_hl_seq.call @fn1(%arg0) : (memref<?xf32>) -> memref<?xf32>
  return %0 : memref<?xf32>
}

// CHECK: @fn1
func @fn1(%arg0 : memref<?xf32>) -> memref<?xf32> {
  // CHECK: iree_hl_seq.call @fn2
  %0 = iree_hl_seq.call @fn2(%arg0) : (memref<?xf32>) -> memref<?xf32>
  return %0 : memref<?xf32>
}

// CHECK: @fn2
func @fn2(%arg0 : memref<?xf32>) -> memref<?xf32> {
  return %arg0 : memref<?xf32>
}

// CHECK-NOT: @unusedFn3
func @unusedFn3(%arg0 : memref<?xf32>) -> memref<?xf32> {
  return %arg0 : memref<?xf32>
}

// -----

// CHECK-NOT: @unusedFn
func @unusedFn(%arg0 : memref<?xf32>) -> memref<?xf32> {
  return %arg0 : memref<?xf32>
}

// -----

// CHECK-LABEL: @exportedFnWithImports
func @exportedFnWithImports(%arg0 : memref<?xf32>) -> memref<?xf32>
    attributes {iree.module.export} {
  // CHECK: iree_hl_seq.call @usedImportFn
  %0 = iree_hl_seq.call @usedImportFn(%arg0) : (memref<?xf32>) -> memref<?xf32>
  return %0 : memref<?xf32>
}

// CHECK: @usedImportFn
func @usedImportFn(%arg0 : memref<?xf32>) -> memref<?xf32>

// CHECK-NOT: @unusedImportFn
func @unusedImportFn(%arg0 : memref<?xf32>) -> memref<?xf32>
