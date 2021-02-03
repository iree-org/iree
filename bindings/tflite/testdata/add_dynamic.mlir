func @main(
    %input : tensor<?xf32> {iree.identifier = "input"}
  ) -> (
    tensor<?xf32> {iree.identifier = "output"}
  ) attributes { iree.module.export } {
  %result = mhlo.add %input, %input : tensor<?xf32>
  return %result : tensor<?xf32>
}
