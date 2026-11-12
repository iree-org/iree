// Argmax over the innermost dimension (value + index outputs), 128x32000 f32.

func.func @argmax_inner_128x32000(%x: tensor<128x32000xf32>) -> tensor<128xi64> {
  %ninf = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : i64
  %ev = tensor.empty() : tensor<128xf32>
  %ei = tensor.empty() : tensor<128xi64>
  %fv = linalg.fill ins(%ninf : f32) outs(%ev : tensor<128xf32>) -> tensor<128xf32>
  %fi = linalg.fill ins(%c0 : i64) outs(%ei : tensor<128xi64>) -> tensor<128xi64>
  %r:2 = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i)>, affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%x : tensor<128x32000xf32>) outs(%fv, %fi : tensor<128xf32>, tensor<128xi64>) {
  ^bb0(%in: f32, %bv: f32, %bi: i64):
    %j = linalg.index 1 : index
    %ji = arith.index_cast %j : index to i64
    %gt = arith.cmpf ogt, %in, %bv : f32
    %nv = arith.select %gt, %in, %bv : f32
    %ni = arith.select %gt, %ji, %bi : i64
    linalg.yield %nv, %ni : f32, i64
  } -> (tensor<128xf32>, tensor<128xi64>)
  return %r#1 : tensor<128xi64>
}
