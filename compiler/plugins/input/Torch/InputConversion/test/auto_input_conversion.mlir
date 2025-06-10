// RUN: iree-compile --compile-to=input --split-input-file %s | FileCheck %s

// Check that the auto input conversion pipeline uses this plugin.

// CHECK-LABEL: util.func public @simple_add_onnx
// CHECK:  arith.addi
func.func @simple_add_onnx(%arg0: !torch.vtensor<[],si64>, %arg1: !torch.vtensor<[],si64>) -> !torch.vtensor<[],si64> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "2.1.0"} {
  %0 = torch.operator "onnx.Add"(%arg0, %arg1) : (!torch.vtensor<[],si64>, !torch.vtensor<[],si64>) -> !torch.vtensor<[],si64>
  return %0 : !torch.vtensor<[],si64>
}

// -----

// Tests that a function using torch types but not containing any ops is still
// handled by the torch input pipeline.

// CHECK: util.func public @nop$async
// CHECK: util.func public @nop(%{{.+}}: !hal.buffer_view) -> !hal.buffer_view
func.func @nop(%arg0: !torch.vtensor<[5],f32>) -> !torch.vtensor<[5],f32> attributes {torch.assume_strict_symbolic_shapes} {
  return %arg0 : !torch.vtensor<[5],f32>
}
