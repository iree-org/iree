// RUN: iree-opt --split-input-file --iree-spirv-vectorize %s | FileCheck %s

func.func @ncw_conv_1d(%input: tensor<2x4x4xf32>, %filter: tensor<4x4x1xf32>, %init: tensor<2x4x4xf32>) -> tensor<2x4x4xf32> {
  %0 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>}
         ins(%input, %filter : tensor<2x4x4xf32>, tensor<4x4x1xf32>)
         outs(%init : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  return %0: tensor<2x4x4xf32>
}

//   CHECK-LABEL: func.func @ncw_conv_1d
//    CHECK-SAME: (%[[INPUT:.+]]: tensor<2x4x4xf32>, %[[FILTER:.+]]: tensor<4x4x1xf32>, %[[INIT:.+]]: tensor<2x4x4xf32>)

//  CHECK-COUNT-8:   vector.transfer_read %[[INPUT]]{{.+}} : tensor<2x4x4xf32>, vector<4xf32>
// CHECK-COUNT-16:   vector.transfer_read %[[FILTER]]{{.+}} : tensor<4x4x1xf32>, vector<1xf32>
//  CHECK-COUNT-8:   vector.transfer_read %[[INIT]]{{.+}} : tensor<2x4x4xf32>, vector<4xf32>
// CHECK-COUNT-16:   vector.extract %{{.+}}[0] : vector<1xf32>
//      CHECK-NOT:   vector.insert
// CHECK-COUNT-32:   vector.fma {{.+}} : vector<4xf32>
//      CHECK-NOT:   vector.insert
//  CHECK-COUNT-8:   vector.transfer_write %{{.+}} : vector<4xf32>, tensor<2x4x4xf32>

