// softmax(matmul(arg0, arg1))
func.func @main(%arg0: tensor<?x16xf32>, %arg1: tensor<16x16xf32>)
    -> tensor<?x16xf32> {
  %c0 = arith.constant 0 : index
  %m = tensor.dim %arg0, %c0 : tensor<?x16xf32>
  %empty = tensor.empty(%m) : tensor<?x16xf32>
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x16xf32>)
               -> tensor<?x16xf32>
  %mat = linalg.matmul ins(%arg0, %arg1 : tensor<?x16xf32>, tensor<16x16xf32>)
                      outs(%init : tensor<?x16xf32>) -> tensor<?x16xf32>
  %out = linalg.softmax dimension(1) ins(%mat : tensor<?x16xf32>)
                        outs(%empty : tensor<?x16xf32>) -> tensor<?x16xf32>
  return %out : tensor<?x16xf32>
}
