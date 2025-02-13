// Minimized from dynamic_tosa_quantized_fully_connected_issue_10859.mlir

func.func @clamp(%arg0: tensor<?xi8>) ->tensor<?xi8>{
  %1 = tosa.clamp %arg0 {max_val = 127 : i8, min_val = -128 : i8} : (tensor<?xi8>) -> tensor<?xi8>
  return %1 : tensor<?xi8>
}
