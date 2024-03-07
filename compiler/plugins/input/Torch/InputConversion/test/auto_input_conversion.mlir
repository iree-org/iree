// RUN: iree-compile --compile-to=input --split-input-file %s | FileCheck %s

// Check that the auto input conversion pipeline uses this plugin.

// CHECK-LABEL: util.func public @simple_add_torch
// CHECK:  arith.addf
func.func @simple_add_torch(%arg0: !torch.vtensor<[2],f32>, %arg1: !torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>, !torch.int -> !torch.vtensor<[2],f32>
  return %0 : !torch.vtensor<[2],f32>
}

// -----

// CHECK-LABEL: util.func public @simple_add_onnx
// CHECK:  arith.addi
func.func @simple_add_onnx(%arg0: !torch.vtensor<[],si64>, %arg1: !torch.vtensor<[],si64>) -> !torch.vtensor<[],si64> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "2.1.0"} {
  %0 = torch.operator "onnx.Add"(%arg0, %arg1) : (!torch.vtensor<[],si64>, !torch.vtensor<[],si64>) -> !torch.vtensor<[],si64>
  return %0 : !torch.vtensor<[],si64>
}
