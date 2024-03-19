// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @dispatchTypes
util.func public @dispatchTypes(
    // CHECK-SAME: %arg0: !flow.dispatch.tensor<readonly:tensor<f32>>
    %arg0: !flow.dispatch.tensor<readonly:tensor<f32>>,
    // CHECK-SAME: %arg1: !flow.dispatch.tensor<readonly:tensor<4x4xf32>>
    %arg1: !flow.dispatch.tensor<readonly:tensor<4x4xf32>>,
    // CHECK-SAME: %arg2: !flow.dispatch.tensor<readonly:tensor<1x2x3x4x5x6xf32>>
    %arg2: !flow.dispatch.tensor<readonly:tensor<1x2x3x4x5x6xf32>>,
    // CHECK-SAME: %arg3: !flow.dispatch.tensor<readonly:tensor<?xf32>>
    %arg3: !flow.dispatch.tensor<readonly:tensor<?xf32>>,
    // CHECK-SAME: %arg4: !flow.dispatch.tensor<readonly:tensor<1x?x3xf32>>
    %arg4: !flow.dispatch.tensor<readonly:tensor<1x?x3xf32>>,
    // CHECK-SAME: %arg5: !flow.dispatch.tensor<writeonly:tensor<f32>>
    %arg5: !flow.dispatch.tensor<writeonly:tensor<f32>>,
    // CHECK-SAME: %arg6: !flow.dispatch.tensor<writeonly:tensor<4x4xf32>>
    %arg6: !flow.dispatch.tensor<writeonly:tensor<4x4xf32>>,
    // CHECK-SAME: %arg7: !flow.dispatch.tensor<writeonly:tensor<1x2x3x4x5x6xf32>>
    %arg7: !flow.dispatch.tensor<writeonly:tensor<1x2x3x4x5x6xf32>>,
    // CHECK-SAME: %arg8: !flow.dispatch.tensor<writeonly:tensor<?xf32>>
    %arg8: !flow.dispatch.tensor<writeonly:tensor<?xf32>>,
    // CHECK-SAME: %arg9: !flow.dispatch.tensor<writeonly:tensor<1x?x3xf32>>
    %arg9: !flow.dispatch.tensor<writeonly:tensor<1x?x3xf32>>
    ) {
  util.return
}
