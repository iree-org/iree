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
// CHECK-SAME: %[[SHAPE:[^:[:space:]]+]]: !shapex.ranked_shape<[1,?,2,?]>
// CHECK-SAME: -> (tensor<1x?x2x?xf32>, !shapex.ranked_shape<[1,?,2,?]>)
func @dynamicFunctionArgs(%arg0 : tensor<1x?x2x?xf32>) -> tensor<1x?x2x?xf32> {
  // Should insert tie on arguments and get_shape on result shapex.
  // CHECK-DAG: %[[TIE_T:.+]] = shapex.tie_shape %[[T]], %[[SHAPE]]
  // CHECK-DAG: %[[GET_SHAPE:.+]] = shapex.get_ranked_shape %[[TIE_T]]
  // CHECK-DAG: return %[[TIE_T]], %[[GET_SHAPE]] : tensor<1x?x2x?xf32>, !shapex.ranked_shape<[1,?,2,?]>
  return %arg0 : tensor<1x?x2x?xf32>
}

// -----
// CHECK-LABEL: @dynamicReturnInBlock
// CHECK-SAME: %[[T:[^:[:space:]]+]]: tensor<2x3xf32>
// CHECK-SAME: -> (tensor<?x3xf32>, !shapex.ranked_shape<[?,3]>)
// Should insert function shape argument and result.
func @dynamicReturnInBlock(%arg0 : tensor<2x3xf32>) -> (tensor<?x3xf32>) {
  %0 = addf %arg0, %arg0 : tensor<2x3xf32>
  br ^bb1
  // CHECK: ^bb1
  ^bb1:
  // CHECK: %[[RESULT:.+]] = "unknown_op"
  %1 = "unknown_op"(%0) : (tensor<2x3xf32>) -> (tensor<?x3xf32>)
  // CHECK: %[[SHAPE:.+]] = shapex.get_ranked_shape %[[RESULT]]
  // CHECK: return %[[RESULT]], %[[SHAPE]] : tensor<?x3xf32>, !shapex.ranked_shape<[?,3]>
  return %1 : tensor<?x3xf32>
}
