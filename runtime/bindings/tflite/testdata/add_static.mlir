func.func @main(
    %input : tensor<1x8x8x3xf32> {iree.abi.name = "input"}
  ) -> (
    tensor<1x8x8x3xf32> {iree.abi.name = "output"}
  ) {
  %result = arith.addf %input, %input : tensor<1x8x8x3xf32>
  return %result : tensor<1x8x8x3xf32>
}
