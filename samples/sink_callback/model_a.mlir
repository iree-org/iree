// softmax(matmul(arg0,weights)) + pi
#map = affine_map<(d0, d1) -> (d0, d1)>
module @foo {
  util.global private @a0 = #flow.parameter.named<"my_scope"::"my_weights"> : tensor<18x18xf32>
  func.func @foo(%arg0: tensor<?x18xf32>) -> tensor<?x18xf32> {
    %a0 = util.global.load @a0 : tensor<18x18xf32>
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x18xf32>
    %0 = tensor.empty(%dim) : tensor<?x18xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x18xf32>) -> tensor<?x18xf32>
    %3 = linalg.matmul ins(%arg0, %a0 : tensor<?x18xf32>, tensor<18x18xf32>) outs(%1 : tensor<?x18xf32>) -> tensor<?x18xf32>
    %4 = linalg.softmax dimension(1) ins(%3 : tensor<?x18xf32>) outs(%1 : tensor<?x18xf32>) -> tensor<?x18xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<?x18xf32>) outs(%0 : tensor<?x18xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_0 = arith.constant 3.1415e+00 : f32 // <--- different constant to model b
      %6 = arith.addf %in, %cst_0 : f32
      linalg.yield %6 : f32
    } -> tensor<?x18xf32>
    return %5 : tensor<?x18xf32>
  }
}
