// RUN: iree-opt -split-input-file -verify-diagnostics -iree-shape-expand-function-dynamic-dims %s | IreeFileCheck %s

// CHECK-LABEL: @staticFunctionArgs
// CHECK-NOT: ranked_shape
func @staticFunctionArgs(%arg0 : tensor<1x2xf32>) {
  return
}

// -----
// CHECK-LABEL: @dynamicFunctionArgs
// Should insert function shape argument and result.
// CHECK-SAME: %[[T:[^:[:space:]]+]]: tensor<1x?x2x?xf32>
// CHECK-SAME: %[[SHAPE:[^:[:space:]]+]]: !shape.ranked_shape<1x?x2x?xi32>
// CHECK-SAME: -> (tensor<1x?x2x?xf32>, !shape.ranked_shape<1x?x2x?xi32>)
func @dynamicFunctionArgs(%arg0 : tensor<1x?x2x?xf32>) -> tensor<1x?x2x?xf32> {
  // Should insert tie on arguments and get_shape on result shape.
  // CHECK-DAG: %[[TIE_T:.+]] = shape.tie_shape %[[T]], %[[SHAPE]]
  // CHECK-DAG: %[[GET_SHAPE:.+]] = shape.get_ranked_shape %[[TIE_T]]
  // CHECK-DAG: return %[[TIE_T]], %[[GET_SHAPE]] : tensor<1x?x2x?xf32>, !shape.ranked_shape<1x?x2x?xi32>
  return %arg0 : tensor<1x?x2x?xf32>
}
