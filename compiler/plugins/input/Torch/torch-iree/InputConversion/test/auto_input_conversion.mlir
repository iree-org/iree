// RUN: iree-compile --compile-to=input --split-input-file %s | FileCheck %s

// Check that the auto input conversion pipeline uses this plugin.

// CHECK-LABEL: func.func @simple_add_torch
// CHECK:  arith.addf
func.func @simple_add_torch(%arg0: !torch.vtensor<[2],f32>, %arg1: !torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>, !torch.int -> !torch.vtensor<[2],f32>
  return %0 : !torch.vtensor<[2],f32>
}
