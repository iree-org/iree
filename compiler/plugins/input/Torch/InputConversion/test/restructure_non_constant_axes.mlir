// RUN: iree-opt -split-input-file \
// RUN:   --torch-onnx-to-torch-backend-pipeline --torch-to-iree %s | FileCheck %s

// Smoke test guarding that RestructureNonConstantAxes is scheduled inside the
// torch-to-iree input pipeline. An ONNX ReduceSum with a *runtime* axes tensor
// (keepdims=1) previously failed with
//   "failed to legalize operation 'torch.aten.sum.dim_IntList'".
// After the pass is scheduled, the runtime axis is statically resolved: the
// input is reshaped and the reduction lowers cleanly to linalg, with no
// torch.aten.sum.dim_IntList surviving.
// (part of iree-org/iree#24378)

// CHECK-LABEL: util.func public @test_reduce_sum_keepdims_random$async
// CHECK-NOT:     torch.aten.sum.dim_IntList
// CHECK:         tensor.reshape
// CHECK:         linalg.fill
// CHECK:         linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "reduction", "parallel"]
func.func @test_reduce_sum_keepdims_random(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %none = torch.constant.none
  %0 = torch.operator "onnx.ReduceSum"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32>
  return %0 : !torch.vtensor<[3,1,2],f32>
}
