module {
  func.func @dense_f32() -> tensor<4xf32> {
    %cst = arith.constant #util.elements<dense_elements_f32> : tensor<4xf32>
    return %cst : tensor<4xf32>
  }
}

{-#
  dialect_resources: {
    util: {
      dense_elements_f32: "0x40000000CDCC8C3FCDCC0C403333534000000000"
    }
  }
#-}
