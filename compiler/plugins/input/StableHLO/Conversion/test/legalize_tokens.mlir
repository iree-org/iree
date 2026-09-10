// RUN: iree-opt --split-input-file --iree-stablehlo-input-transformation-pipeline %s \
// RUN:   | FileCheck %s --implicit-check-not=stablehlo.

// CHECK-LABEL: @create_token_chain
// CHECK: return %{{.+}} : tensor<4xf32>
func.func @create_token_chain(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %token = stablehlo.create_token : !stablehlo.token
  %joined = "stablehlo.after_all"(%token) : (!stablehlo.token) -> !stablehlo.token
  %0 = stablehlo.abs %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// A token in the signature becomes a placeholder tensor, with the original
// type recorded for the ABI.
// CHECK-LABEL: @token_in_signature
// CHECK-SAME: %[[ARG0:.+]]: tensor<i1> {iree.abi.encoding = !stablehlo.token}
// CHECK-SAME: -> (tensor<i1> {iree.abi.encoding = !stablehlo.token})
// CHECK: return %[[ARG0]] : tensor<i1>
func.func @token_in_signature(%arg0: !stablehlo.token) -> !stablehlo.token {
  %0 = "stablehlo.after_all"(%arg0) : (!stablehlo.token) -> !stablehlo.token
  return %0 : !stablehlo.token
}
