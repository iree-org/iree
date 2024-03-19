func.func @main(
    %input : tensor<2xf32> {iree.abi.name = "input"}
  ) -> (
    tensor<2xf32> {iree.abi.name = "output"}
  ) {
  %result = arith.addf %input, %input : tensor<2xf32>
  return %result : tensor<2xf32>
}
