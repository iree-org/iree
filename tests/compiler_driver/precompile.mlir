// RUN: iree-compile --compile-mode=precompile --iree-hal-target-backends=vmvx %s | iree-compile --iree-hal-target-backends=vmvx --compile-from=global-optimization --compile-to=stream - > %t && \
// RUN: iree-compile --iree-hal-target-backends=vmvx --compile-to=stream %s | diff - %t

func.func @test(%arg0 : tensor<10x20xf32>, %arg1 : tensor<20x30xf32>, %arg2 : tensor<10x30xf32>) -> tensor<10x30xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<10x20xf32>, tensor<20x30xf32>)
      outs(%arg2 : tensor<10x30xf32>) -> tensor<10x30xf32>
  return %0 : tensor<10x30xf32>
}
