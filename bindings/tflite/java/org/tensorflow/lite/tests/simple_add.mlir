func @main(
    %input : tensor<2xf32> {iree.identifier = "input"}
  ) -> (
    tensor<2xf32> {iree.identifier = "output"}
  ) attributes { iree.module.export } {
  %result = mhlo.add %input, %input : tensor<2xf32>
  return %result : tensor<2xf32>
}
