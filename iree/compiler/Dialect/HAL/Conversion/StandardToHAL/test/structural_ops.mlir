// RUN: iree-opt -split-input-file -iree-convert-to-hal %s | IreeFileCheck %s

// CHECK-LABEL: func @tensorIO(%arg0: !hal.buffer) -> !hal.buffer
func @tensorIO(%arg0 : tensor<1x1xi32>) -> tensor<1x1xi32> {
  // CHECK-NEXT: br ^bb1(%arg0 : !hal.buffer)
  br ^bb1(%arg0 : tensor<1x1xi32>)
// CHECK-NEXT: ^bb1(%[[BB0:.+]]: !hal.buffer)
^bb1(%0 : tensor<1x1xi32>):
  // CHECK-NEXT: return %[[BB0]] : !hal.buffer
  return %0 : tensor<1x1xi32>
}

// -----

// CHECK-LABEL: func @select(%arg0: i1, %arg1: !hal.buffer, %arg2: !hal.buffer) -> !hal.buffer
func @select(%arg0: i1, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
  // CHECK: %[[RES:.+]] = select %arg0, %arg1, %arg2 : !hal.buffer
  %0 = select %arg0, %arg1, %arg2 : tensor<i32>
  // CHECK: return %[[RES]] : !hal.buffer
  return %0 : tensor<i32>
}

// -----
// CHECK-LABEL: func @tensor_cast_does_not_alias_metadata_update
func @tensor_cast_does_not_alias_metadata_update(%arg0: !hal.buffer_view) -> !hal.buffer_view {
    %0 = hal.tensor.cast %arg0 : !hal.buffer_view -> tensor<3x2x2x2xf32>
    %1 = flow.ex.stream.fragment(%0) : (tensor<3x2x2x2xf32>) -> (tensor<3x2x1x4x1xf32>) =
        (%arg1: tensor<3x2x2x2xf32>) -> tensor<3x2x1x4x1xf32> {
      %3 = flow.tensor.reshape %arg1 : tensor<3x2x2x2xf32> -> tensor<3x2x1x4x1xf32>
      flow.return %3 : tensor<3x2x1x4x1xf32>
    }

    // Just anchor on the end of the function that creates a new buffer view.
    // CHECK: hal.ex.submit_and_wait
    // CHECK: %[[C3:.*]] = constant 3 : index
    // CHECK: %[[C2:.*]] = constant 2 : index
    // CHECK: %[[C1_1:.*]] = constant 1 : index
    // CHECK: %[[C4:.*]] = constant 4 : index
    // CHECK: %[[C1_2:.*]] = constant 1 : index
    // CHECK: %[[ET:.*]] = constant 50331680 : i32
    // CHECK: %[[VIEW:.*]] = hal.buffer_view.create
    //   CHECK-SAME: shape([%[[C3]], %[[C2]], %[[C1_1]], %[[C4]], %[[C1_2]]])
    //   CHECK-SAME: type(%[[ET]])
    // CHECK: return %[[VIEW]]
    %2 = hal.tensor.cast %1 : tensor<3x2x1x4x1xf32> -> !hal.buffer_view
    return %2 : !hal.buffer_view
}
