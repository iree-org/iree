// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @dispatchTypes
func.func @dispatchTypes(
    // CHECK-SAME: %arg0: !flow.dispatch.tensor<readonly:f32>
    %arg0: !flow.dispatch.tensor<readonly:f32>,
    // CHECK-SAME: %arg1: !flow.dispatch.tensor<readonly:4x4xf32>
    %arg1: !flow.dispatch.tensor<readonly:4x4xf32>,
    // CHECK-SAME: %arg2: !flow.dispatch.tensor<readonly:1x2x3x4x5x6xf32>
    %arg2: !flow.dispatch.tensor<readonly:1x2x3x4x5x6xf32>,
    // CHECK-SAME: %arg3: !flow.dispatch.tensor<readonly:?xf32>
    %arg3: !flow.dispatch.tensor<readonly:?xf32>,
    // CHECK-SAME: %arg4: !flow.dispatch.tensor<readonly:1x?x3xf32>
    %arg4: !flow.dispatch.tensor<readonly:1x?x3xf32>,
    // CHECK-SAME: %arg5: !flow.dispatch.tensor<writeonly:f32>
    %arg5: !flow.dispatch.tensor<writeonly:f32>,
    // CHECK-SAME: %arg6: !flow.dispatch.tensor<writeonly:4x4xf32>
    %arg6: !flow.dispatch.tensor<writeonly:4x4xf32>,
    // CHECK-SAME: %arg7: !flow.dispatch.tensor<writeonly:1x2x3x4x5x6xf32>
    %arg7: !flow.dispatch.tensor<writeonly:1x2x3x4x5x6xf32>,
    // CHECK-SAME: %arg8: !flow.dispatch.tensor<writeonly:?xf32>
    %arg8: !flow.dispatch.tensor<writeonly:?xf32>,
    // CHECK-SAME: %arg9: !flow.dispatch.tensor<writeonly:1x?x3xf32>
    %arg9: !flow.dispatch.tensor<writeonly:1x?x3xf32>
    ) {
  return
}
