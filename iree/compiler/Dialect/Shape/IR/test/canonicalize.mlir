// RUN: iree-opt -split-input-file -verify-diagnostics -canonicalize %s | IreeFileCheck %s


// CHECK-LABEL: @elideTiedGetRankedShape
// CHECK-SAME: %[[T:[^:[:space:]]+]]: tensor<1x?x2x?xf32>
// CHECK-SAME: %[[SHAPE:[^:[:space:]]+]]: !shape.ranked_shape<1x?x2x?xi32>
func @elideTiedGetRankedShape(%arg0: tensor<1x?x2x?xf32>, %arg1: !shape.ranked_shape<1x?x2x?xi32>) -> (tensor<1x?x2x?xf32>, !shape.ranked_shape<1x?x2x?xi32>) {
  // Note that canonicalization does *not* remove tie_shape. That must be
  // removed manually once all shape materialization is complete (otherwise,
  // information needed to materialize would be lost).
  // CHECK: %[[TIE_T:.+]] = shape.tie_shape %[[T]], %[[SHAPE]]
  %0 = shape.tie_shape %arg0, %arg1 : tensor<1x?x2x?xf32>, !shape.ranked_shape<1x?x2x?xi32>
  // CHECK-NOT: shape.get_ranked_shape
  %1 = shape.get_ranked_shape %0 : tensor<1x?x2x?xf32> -> !shape.ranked_shape<1x?x2x?xi32>
  // CHECK-DAG: return %[[TIE_T]], %[[SHAPE]]
  return %0, %1 : tensor<1x?x2x?xf32>, !shape.ranked_shape<1x?x2x?xi32>
}
