// Minimized from dynamic_tosa_quantized_fully_connected_issue_10859.mlir

func.func @clamp(%arg0: tensor<?xi8>) ->tensor<?xi8>{
  %1 = "tosa.clamp"(%arg0) {max_fp = 0.000000e+00 : f32, max_int = 127 : i64, min_fp = 0.000000e+00 : f32, min_int = -128 : i64} : (tensor<?xi8>) -> tensor<?xi8>
  return %1 : tensor<?xi8>
}
