// RUN: iree-compile --iree-input-type=stablehlo --iree-hal-target-backends=cuda --iree-cuda-target=sm_89 %s -o /dev/null
// RUN: iree-compile --iree-input-type=stablehlo --iree-hal-target-backends=cuda --iree-cuda-target=rtx4090 %s -o /dev/null

// Checks that sm_89 and RTX 40 aliases are accepted by the CUDA target.
module {
  func.func @main(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}
