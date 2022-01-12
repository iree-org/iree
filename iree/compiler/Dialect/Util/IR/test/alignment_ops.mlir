// RUN: iree-opt -split-input-file %s | FileCheck %s

// CHECK-LABEL: @utilAlign
func @utilAlign(%arg0 : index, %arg1: index) {
  // CHECK: = util.align %arg0, %arg1 : index
  %result = util.align %arg0, %arg1 : index
  return
}

// -----

// CHECK-LABEL: @utilAlignInt
func @utilAlignInt(%arg0 : i32, %arg1: i32) {
  // CHECK: = util.align %arg0, %arg1 : i32
  %result = util.align %arg0, %arg1 : i32
  return
}
