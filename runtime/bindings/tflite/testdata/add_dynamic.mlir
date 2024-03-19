func.func @main(
    %input : tensor<?xf32> {iree.abi.name = "input"}
  ) -> (
    tensor<?xf32> {iree.abi.name = "output"}
  ) {
  %result = arith.addf %input, %input : tensor<?xf32>
  return %result : tensor<?xf32>
}
