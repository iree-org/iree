// Tests folding and canonicalization of stream ops.

// RUN: iree-opt -split-input-file -canonicalize %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: func @inlineConstant
func @inlineConstant() -> index {
  %cst = constant 4 : index
  // CHECK: flow.ex.stream.fragment()
  %0 = flow.ex.stream.fragment(%cst) : (index) -> index =
      (%arg0: index) -> index {
    // CHECK: %[[C:.+]] = constant 4 : index
    // CHECK-NEXT: return %[[C]]
    flow.return %arg0 : index
  }
  return %0 : index
}

// -----

// CHECK-LABEL: func @removeUnusedCapture
// CHECK-SAME: (%[[ARG:.+]]: index)
func @removeUnusedCapture(%arg: index) -> index {
  %unused = constant 5 : index
  // CHECK: flow.ex.stream.fragment(%[[ARG]])
  %0 = flow.ex.stream.fragment(%arg, %unused) : (index, index) -> index =
      (%arg0: index, %arg1: index) -> index {
    flow.return %arg0 : index
  }
  return %0 : index
}

// -----

// CHECK-LABEL: func @removeUnusedDupCapture
// CHECK-SAME: (%[[ARG:.+]]: index)
func @removeUnusedDupCapture(%arg: index) -> index {
  // CHECK: flow.ex.stream.fragment(%[[ARG]])
  %0 = flow.ex.stream.fragment(%arg, %arg) : (index, index) -> index =
      (%arg0: index, %arg1: index) -> index {
    flow.return %arg1 : index
  }
  return %0 : index
}

// -----

// CHECK-LABEL: func @removeUnusedResult
// CHECK-SAME: (%[[ARG0:.+]]: index, %[[ARG1:.+]]: index)
func @removeUnusedResult(%arg0: index, %arg1: index) -> index {
  // CHECK: flow.ex.stream.fragment(%[[ARG1]])
  %0:2 = flow.ex.stream.fragment(%arg0, %arg1) : (index, index) -> (index, index) =
      (%arg0: index, %arg1: index) -> (index, index) {
    flow.return %arg1, %arg0 : index, index
  }
  return %0#0 : index
}
