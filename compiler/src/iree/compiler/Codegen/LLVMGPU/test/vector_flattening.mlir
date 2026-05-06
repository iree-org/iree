// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmgpu-vector-flattening))" \
// RUN:   --split-input-file %s | FileCheck %s

// CHECK-LABEL: func @vector_multi_reduction_flattening
// CHECK-SAME:   %[[INPUT:.+]]: vector<2x4xf32>, %[[ACC:.*]]: f32)
func.func @vector_multi_reduction_flattening(%arg0: vector<2x4xf32>, %acc: f32) -> f32 {
    // CHECK: %[[CASTED:.*]] = vector.shape_cast %[[INPUT]] : vector<2x4xf32> to vector<8xf32>
    // CHECK: %[[RESULT:.+]] = vector.multi_reduction <mul>, %[[CASTED]], %[[ACC]] [0]
    %0 = vector.multi_reduction <mul>, %arg0, %acc [0, 1] : vector<2x4xf32> to f32
    // CHECK: return %[[RESULT]]
    return %0 : f32
}
