func.func @i1_type() {
  %c0 = arith.constant 0 : index
  %c255 = arith.constant 255 : i8
  %input1 = util.unfoldable_constant dense<[85]> : tensor<1xi8>  // b01010101
  %input2 = util.unfoldable_constant dense<[170]> : tensor<1xi8> // b10101010
  %lhs = flow.tensor.bitcast %input1 : tensor<1xi8> -> tensor<8xi1>
  %rhs = flow.tensor.bitcast %input2 : tensor<1xi8> -> tensor<8xi1>
  %empty = tensor.empty() : tensor<8xi1>
  %res = linalg.generic
        {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
        ins(%lhs, %rhs : tensor<8xi1>, tensor<8xi1>) outs(%empty: tensor<8xi1>) {
  ^bb0(%inlhs: i1, %inrhs: i1, %out: i1):
    %inres = arith.xori %inlhs, %inrhs: i1
    linalg.yield %inres : i1
  } -> tensor<8xi1>
  %tensor_res = flow.tensor.bitcast %res : tensor<8xi1> -> tensor<1xi8>
  check.expect_eq_const(%tensor_res, dense<[255]> : tensor<1xi8>) : tensor<1xi8>
  return
}

func.func @i1_type_slice() {
  %input = util.unfoldable_constant dense<[0, 255, 0]> : tensor<3xi8>
  %flat_input_all = flow.tensor.bitcast %input : tensor<3xi8> -> tensor<24xi1>
  %slice = tensor.extract_slice %flat_input_all[8][8][1] : tensor<24xi1> to tensor<8xi1>
  %tensor_res = flow.tensor.bitcast %slice : tensor<8xi1> -> tensor<1xi8>
  check.expect_eq_const(%tensor_res, dense<[255]> : tensor<1xi8>) : tensor<1xi8>
  return
}
