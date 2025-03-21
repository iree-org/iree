// RUN: iree-compile --compile-mode=precompile --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | FileCheck %s

func.func @test(%arg0 : tensor<10x20xf32>, %arg1 : tensor<20x30xf32>, %arg2 : tensor<10x30xf32>) -> tensor<10x30xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<10x20xf32>, tensor<20x30xf32>)
      outs(%arg2 : tensor<10x30xf32>) -> tensor<10x30xf32>
  return %0 : tensor<10x30xf32>
}

// Just check that we have the right target and executable targets.
//      CHECK: module
// CHECK-SAME: stream.affinity.default = #hal.device.affinity<@[[DEVICE:.+]]>
//      CHECK: util.global private @[[DEVICE]] = #hal.device.target<"local", [#hal.executable.target<"vmvx"
