// RUN: iree-opt -split-input-file --iree-codegen-broadcast-in-dim-to-reshape %s | IreeFileCheck %s
func @broadcast_in_dim_to_reshape(%arg0: tensor<1000xf32>) -> tensor<1x1x1x1000xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<1000xf32>) -> tensor<1x1x1x1000xf32>
    return %0 : tensor<1x1x1x1000xf32> 
}
// CHECK: func @broadcast_in_dim_to_reshape(%[[ARG0:.+]]: tensor<1000xf32>)
// CHECK: "mhlo.reshape"(%[[ARG0]]) : (tensor<1000xf32>) -> tensor<1x1x1x1000xf32> 
