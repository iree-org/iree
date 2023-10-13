// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-codegen-generic-vectorization,iree-spirv-initial-vector-lowering,iree-codegen-hoist-redundant-vector-transfers,iree-spirv-final-vector-lowering))' \
// RUN:   %s | FileCheck %s

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
// CHECK-COUNT-16:   vector.extract %{{.+}}[0] : f32 from vector<1xf32>
//      CHECK-NOT:   vector.insert
// CHECK-COUNT-32:   vector.fma {{.+}} : vector<4xf32>
//      CHECK-NOT:   vector.insert
//  CHECK-COUNT-8:   vector.transfer_write %{{.+}} : vector<4xf32>, tensor<2x4x4xf32>

// -----

// Check that emit SPIR-V integer dot product instructions when supported by
// the target env. We expect the conv to follow the inner product lowering.

func.func @nwc_conv_1d_dot_prod(%input: tensor<1x7x3xi8>, %filter: tensor<1x3x4xi8>) -> tensor<1x4x4xi32> attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.5,
                                         [DotProduct, DotProductInputAll, DotProductInput4x8Bit],
                                         [SPV_KHR_integer_dot_product]>,
                                       #spirv.resource_limits<>> } {
  %c0 = arith.constant 0 : i32
  %i0 = arith.constant 0 : index
  %init = tensor.empty() : tensor<1x4x4xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%init : tensor<1x4x4xi32>) -> tensor<1x4x4xi32>
  %conv = linalg.conv_1d_nwc_wcf {
    dilations = dense<1> : vector<1xi64>,
    strides = dense<2> : vector<1xi64>
  }
  ins(%input, %filter : tensor<1x7x3xi8>, tensor<1x3x4xi8>)
  outs(%fill : tensor<1x4x4xi32>) -> tensor<1x4x4xi32>
  return %conv: tensor<1x4x4xi32>
}

//    CHECK-LABEL: func.func @nwc_conv_1d_dot_prod
//          CHECK:   %[[ZERO:.+]] = spirv.Constant 0 : i8

//          CHECK:   %[[LHS:.+]] = spirv.CompositeConstruct %{{.+}}, %[[ZERO]] : (vector<3xi8>, i8) -> vector<4xi8>
//          CHECK:   %[[RHS:.+]] = spirv.CompositeConstruct %{{.+}}, %[[ZERO]] : (vector<3xi8>, i8) -> vector<4xi8>
//          CHECK:   spirv.SDotAccSat %[[LHS]], %[[RHS]], %{{.+}} : (vector<4xi8>, vector<4xi8>, i32) -> i32
//  CHECK-COUNT-2:   spirv.CompositeConstruct %{{.+}}, %[[ZERO]] : (vector<3xi8>, i8) -> vector<4xi8>
//          CHECK:   spirv.SDotAccSat %{{.+}}, %{{.+}}, %{{.+}} : (vector<4xi8>, vector<4xi8>, i32) -> i32
//  CHECK-COUNT-2:   spirv.CompositeConstruct %{{.+}}, %[[ZERO]] : (vector<3xi8>, i8) -> vector<4xi8>
//          CHECK:   spirv.SDotAccSat %{{.+}}, %{{.+}}, %{{.+}} : (vector<4xi8>, vector<4xi8>, i32) -> i32
//  CHECK-COUNT-2:   spirv.CompositeConstruct %{{.+}}, %[[ZERO]] : (vector<3xi8>, i8) -> vector<4xi8>
//          CHECK:   spirv.SDotAccSat %{{.+}}, %{{.+}}, %{{.+}} : (vector<4xi8>, vector<4xi8>, i32) -> i32

// CHECK-COUNT-12:   spirv.SDotAccSat {{.+}} : (vector<4xi8>, vector<4xi8>, i32) -> i32
