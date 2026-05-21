// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmgpu-vector-multi-reduction-lowering))" --split-input-file %s | FileCheck %s --check-prefixes=ALL

// ALL-LABEL: func @one_dim_reduction
// ALL-SAME:    %[[INPUT:.+]]: vector<8xf32>, %[[ACC:.+]]: f32
func.func @one_dim_reduction(%arg0: vector<8xf32>, %acc: f32) -> f32 {
  // ALL: %[[RESULT:.+]] = vector.reduction <add>, %[[INPUT]], %[[ACC]] : vector<8xf32> into f32
  %0 = vector.multi_reduction <add>, %arg0, %acc [0] : vector<8xf32> to f32
  // ALL: return %[[RESULT]]
  return %0 : f32
}
