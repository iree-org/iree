// RUN: iree-opt -split-input-file -verify-diagnostics %s

func @sort_invalid_dimension(%arg0: tensor<128xi32>) -> tensor<128xi32> {
  // expected-error @+1 {{dimension must be within (0, 1]}}
  %0 = linalg_ext.sort {dimension = 1 : i64}
    outs(%arg0 : tensor<128xi32>) {
  ^bb0(%arg1: i32, %arg2: i32):  // no predecessors
    %1 = cmpi sgt, %arg1, %arg2 : i32
    linalg_ext.yield %1 : i1
  } -> tensor<128xi32>
  return %0 : tensor<128xi32>
}

// -----

func @sort_without_dimension(%arg0: tensor<3x4xi32>) -> tensor<3x4xi32> {
  // expected-error @+1 {{dimension must be specified if rank > 1}}
  %0 = linalg_ext.sort
    outs(%arg0 : tensor<3x4xi32>) {
  ^bb0(%arg1: i32, %arg2: i32):  // no predecessors
    %1 = cmpi sgt, %arg1, %arg2 : i32
    linalg_ext.yield %1 : i1
  } -> tensor<3x4xi32>
  return %0 : tensor<3x4xi32>
}
