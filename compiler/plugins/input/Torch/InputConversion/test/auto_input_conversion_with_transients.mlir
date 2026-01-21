// RUN: iree-compile --iree-externalize-transients --compile-to=input --split-input-file %s | FileCheck %s

// Check that the auto input conversion respects the driver option to externalize transients.

// CHECK-LABEL: @check_transients_generation
//       CHECK: util.func public @main$async(
//  CHECK-SAME: %arg0: !hal.buffer_view, %arg1: !hal.buffer,
//  CHECK-SAME: %arg2: !hal.fence, %arg3: !hal.fence) -> !hal.buffer_view
//       CHECK: %[[TRANSIENTS_CALL:.+]] = hal.tensor.transients %[[COMP:.+]] : tensor<5x4xf32> from %arg1 : !hal.buffer
//       CHECK: %[[BARRIER:.+]] = hal.tensor.barrier join(%[[TRANSIENTS_CALL]] : tensor<5x4xf32>)

//       CHECK: util.func public @main(%arg0: !hal.buffer_view, %arg1: !hal.buffer) -> !hal.buffer_view
builtin.module @check_transients_generation{
func.func @main(%arg0: !torch.vtensor<[5,4],f32>) -> (!torch.vtensor<[5,4],f32>) {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg0, %int1 : !torch.vtensor<[5,4],f32>, !torch.vtensor<[5,4],f32>, !torch.int -> !torch.vtensor<[5,4],f32>
  return %0 : !torch.vtensor<[5,4],f32>
}
}


// -----
// CHECK-LABEL: @return_immutable_arg
// CHECK: util.func public @main$async
// CHECK-SAME: !hal.buffer
// CHECK: hal.fence.signal<%arg3 : !hal.fence>
// CHECK: util.return %arg0
// CHECK-NOT: hal.tensor.transients
builtin.module @return_immutable_arg {
func.func @main(%arg0: !torch.vtensor<[4,5],si32>) -> !torch.vtensor<[4,5],si32>  {
  return %arg0 : !torch.vtensor<[4,5],si32>
}
}
