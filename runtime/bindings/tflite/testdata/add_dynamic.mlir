func.func @main(
    %input : tensor<?xf32> {iree.identifier = "input"}
  ) -> (
    tensor<?xf32> {iree.identifier = "output"}
  ) {
  %result = arith.addf %input, %input : tensor<?xf32>
  return %result : tensor<?xf32>
}
