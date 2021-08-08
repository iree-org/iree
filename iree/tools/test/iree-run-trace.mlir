// RUN: (iree-translate -iree-input-type=tosa -iree-hal-target-backends=vmvx -iree-mlir-to-vm-bytecode-module %s | iree-run-trace --driver=vmvx iree/tools/test/iree-run-trace.yaml) | IreeFileCheck %s

//      CHECK: --- CALL[module.simple_mul] ---
// CHECK-NEXT: 4xf32=10 4 6 8
// CHECK-NEXT: --- CALL[module.simple_mul] ---
// CHECK-NEXT: 4xf32=4 10 18 28

func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "tosa.mul"(%arg0, %arg1) {shift = 0 : i32} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
