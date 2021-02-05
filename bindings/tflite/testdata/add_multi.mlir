func @main(
    %arg0: tensor<1x8x8x3xf32> {iree.identifier = "a"},
    %arg1: tensor<1x8x8x3xf32> {iree.identifier = "b"},
    %arg2: tensor<1x8x8x3xf32> {iree.identifier = "c"},
    %arg3: tensor<1x8x8x3xf32> {iree.identifier = "d"}
  ) -> (
    tensor<1x8x8x3xf32> {iree.identifier = "x"},
    tensor<1x8x8x3xf32> {iree.identifier = "y"}
  ) attributes { iree.module.export } {
  %0 = "mhlo.add"(%arg1, %arg2) : (tensor<1x8x8x3xf32>, tensor<1x8x8x3xf32>) -> tensor<1x8x8x3xf32>
  %1 = "mhlo.add"(%arg0, %0) : (tensor<1x8x8x3xf32>, tensor<1x8x8x3xf32>) -> tensor<1x8x8x3xf32>
  %2 = "mhlo.add"(%arg3, %0) : (tensor<1x8x8x3xf32>, tensor<1x8x8x3xf32>) -> tensor<1x8x8x3xf32>
  return %1, %2 : tensor<1x8x8x3xf32>, tensor<1x8x8x3xf32>
}
