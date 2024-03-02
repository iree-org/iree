func.func @main(
    %arg0: tensor<1x8x8x3xf32> {iree.abi.name = "a"},
    %arg1: tensor<1x8x8x3xf32> {iree.abi.name = "b"},
    %arg2: tensor<1x8x8x3xf32> {iree.abi.name = "c"},
    %arg3: tensor<1x8x8x3xf32> {iree.abi.name = "d"}
  ) -> (
    tensor<1x8x8x3xf32> {iree.abi.name = "x"},
    tensor<1x8x8x3xf32> {iree.abi.name = "y"}
  ) {
  %0 = arith.addf %arg1, %arg2 : tensor<1x8x8x3xf32>
  %1 = arith.addf %arg0, %0 : tensor<1x8x8x3xf32>
  %2 = arith.addf %arg3, %0 : tensor<1x8x8x3xf32>
  return %1, %2 : tensor<1x8x8x3xf32>, tensor<1x8x8x3xf32>
}
