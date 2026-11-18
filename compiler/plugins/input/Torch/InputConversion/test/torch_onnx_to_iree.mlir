// RUN: iree-opt -split-input-file \
// RUN:   --torch-onnx-to-torch-backend-pipeline --torch-to-iree %s | FileCheck %s

// Tests lowering of ONNX operations through the torch-onnx-to-iree input
// pipeline. New ONNX-to-IREE lowering cases can be added below.

// An ONNX reduction (keepdims=1) with a dynamically-sized axes tensor (1D) can be lowered.
// CHECK-LABEL: util.func public @test_reduce_sum_keepdims$async
// CHECK-NOT:     torch.aten.sum.dim_IntList
// CHECK:         tensor.reshape
// CHECK:         linalg.fill
// CHECK:         linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "reduction", "parallel"]
func.func @test_reduce_sum_keepdims(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %none = torch.constant.none
  %0 = torch.operator "onnx.ReduceSum"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32>
  return %0 : !torch.vtensor<[3,1,2],f32>
}
