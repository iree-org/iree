// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @rangeMin
util.func public @rangeMin(%arg0: index, %arg1: index, %arg2: index) {
  // CHECK: = util.range.min %arg0 : index
  %0 = util.range.min %arg0 : index
  // CHECK: = util.range.min %arg0, %arg1, %arg2 : index
  %1 = util.range.min %arg0, %arg1, %arg2 : index
  util.return
}

// -----

// CHECK-LABEL: @rangeMax
util.func public @rangeMax(%arg0: index, %arg1: index, %arg2: index) {
  // CHECK: = util.range.max %arg0 : index
  %0 = util.range.max %arg0 : index
  // CHECK: = util.range.max %arg0, %arg1, %arg2 : index
  %1 = util.range.max %arg0, %arg1, %arg2 : index
  util.return
}

// -----

// CHECK-LABEL: @rangeExtents
util.func public @rangeExtents(%arg0: index, %arg1: index, %arg2: index) {
  // CHECK: = util.range.extents [%arg0 for %arg2] : index
  %0:2 = util.range.extents [%arg0 for %arg2] : index
  // CHECK: = util.range.extents [%arg0 for %arg2], [%arg1 for %arg2] : index
  %1:2 = util.range.extents [%arg0 for %arg2], [%arg1 for %arg2] : index
  util.return
}
