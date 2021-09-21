// RUN: iree-opt -allow-unregistered-dialect -split-input-file -verify-diagnostics -iree-shape-expand-function-dynamic-dims %s | IreeFileCheck %s

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

// -----
// CHECK-LABEL:   func @calls(
// CHECK-SAME:                %[[ARG:.*]]: tensor<1x?x2x?xf32>,
// CHECK-SAME:                %[[ARG_SHAPE:.*]]: !shapex.ranked_shape<[1,?,2,?]>) -> (tensor<1x?x2x?xf32>, !shapex.ranked_shape<[1,?,2,?]>) {
func @calls(%arg0 : tensor<1x?x2x?xf32>) -> tensor<1x?x2x?xf32> {
  // CHECK:           %[[ARG_TIED:.*]] = shapex.tie_shape %[[ARG]], %[[ARG_SHAPE]] : tensor<1x?x2x?xf32>, !shapex.ranked_shape<[1,?,2,?]>
  // CHECK:           %[[ARG_TIED_SHAPE:.*]] = shapex.get_ranked_shape %[[ARG_TIED]] : tensor<1x?x2x?xf32> -> !shapex.ranked_shape<[1,?,2,?]>
  // CHECK:           %[[CALL_RESULT:.*]]:2 = call @calls(%[[ARG_TIED]], %[[ARG_TIED_SHAPE]]) : (tensor<1x?x2x?xf32>, !shapex.ranked_shape<[1,?,2,?]>) -> (tensor<1x?x2x?xf32>, !shapex.ranked_shape<[1,?,2,?]>)
  %0 = std.call @calls(%arg0) : (tensor<1x?x2x?xf32>) -> tensor<1x?x2x?xf32>
  // CHECK:           %[[CALL_RESULT_TIED:.*]] = shapex.tie_shape %[[CALL_RESULT]]#0, %[[CALL_RESULT]]#1 : tensor<1x?x2x?xf32>, !shapex.ranked_shape<[1,?,2,?]>
  // CHECK:           %[[CALL_RESULT_SHAPE:.*]] = shapex.get_ranked_shape %[[CALL_RESULT_TIED]] : tensor<1x?x2x?xf32> -> !shapex.ranked_shape<[1,?,2,?]>
  // CHECK:           return %[[CALL_RESULT_TIED]], %[[CALL_RESULT_SHAPE]] : tensor<1x?x2x?xf32>, !shapex.ranked_shape<[1,?,2,?]>
  return %0 : tensor<1x?x2x?xf32>
}
