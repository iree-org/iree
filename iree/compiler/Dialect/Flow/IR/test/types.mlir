// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @dispatchTypes
func @dispatchTypes(
    // CHECK-SAME: %arg0: !flow.dispatch.input<f32>
    %arg0: !flow.dispatch.input<f32>,
    // CHECK-SAME: %arg1: !flow.dispatch.input<4x4xf32>
    %arg1: !flow.dispatch.input<4x4xf32>,
    // CHECK-SAME: %arg2: !flow.dispatch.input<1x2x3x4x5x6xf32>
    %arg2: !flow.dispatch.input<1x2x3x4x5x6xf32>,
    // CHECK-SAME: %arg3: !flow.dispatch.input<?xf32>
    %arg3: !flow.dispatch.input<?xf32>,
    // CHECK-SAME: %arg4: !flow.dispatch.input<1x?x3xf32>
    %arg4: !flow.dispatch.input<1x?x3xf32>,
    // CHECK-SAME: %arg5: !flow.dispatch.output<f32>
    %arg5: !flow.dispatch.output<f32>,
    // CHECK-SAME: %arg6: !flow.dispatch.output<4x4xf32>
    %arg6: !flow.dispatch.output<4x4xf32>,
    // CHECK-SAME: %arg7: !flow.dispatch.output<1x2x3x4x5x6xf32>
    %arg7: !flow.dispatch.output<1x2x3x4x5x6xf32>,
    // CHECK-SAME: %arg8: !flow.dispatch.output<?xf32>
    %arg8: !flow.dispatch.output<?xf32>,
    // CHECK-SAME: %arg9: !flow.dispatch.output<1x?x3xf32>
    %arg9: !flow.dispatch.output<1x?x3xf32>
    ) {
  return
}
