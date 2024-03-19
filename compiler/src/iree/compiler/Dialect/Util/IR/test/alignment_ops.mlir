// RUN: iree-opt --split-input-file %s | FileCheck %s

// CHECK-LABEL: @utilAlign
util.func public @utilAlign(%arg0 : index, %arg1: index) {
  // CHECK: = util.align %arg0, %arg1 : index
  %result = util.align %arg0, %arg1 : index
  util.return
}

// -----

// CHECK-LABEL: @utilAlignInt
util.func public @utilAlignInt(%arg0 : i32, %arg1: i32) {
  // CHECK: = util.align %arg0, %arg1 : i32
  %result = util.align %arg0, %arg1 : i32
  util.return
}

// -----

// CHECK-LABEL: @sizeofUnfoldable
util.func public @sizeofUnfoldable() -> index {
  // CHECK: = util.sizeof index
  %0 = util.sizeof index
  util.return %0 : index
}
