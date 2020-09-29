// RUN: iree-opt -split-input-file -iree-vmla-pre-conversion-lowering -iree-vmla-conversion -canonicalize %s | IreeFileCheck %s

func @sort1D(%arg0 : tensor<4xf32>) -> tensor<4xf32> attributes { sym_visibility = "private" } {
  %sort = "mhlo.sort"(%arg0) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
    %compare = "mhlo.compare"(%arg1, %arg2) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%compare) : (tensor<i1>) -> ()
  }) {dimension = 0 : i64, is_stable = false} : (tensor<4xf32>) -> tensor<4xf32>

  return %sort : tensor<4xf32>
}


// CHECK-LABEL: func @sort2D
func @sort2D(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> attributes { sym_visibility = "private" } {
  %sort = "mhlo.sort"(%arg0) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
    %compare = "mhlo.compare"(%arg1, %arg2) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%compare) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = false} : (tensor<4x4xf32>) -> tensor<4x4xf32>

  return %sort : tensor<4x4xf32>
}
