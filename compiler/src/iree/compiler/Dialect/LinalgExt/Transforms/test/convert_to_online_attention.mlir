// RUN: iree-opt --split-input-file --iree-linalg-ext-convert-attention-to-online-attention %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>

func.func @attention(%q: tensor<2x10x4096x128xf16>, %k: tensor<2x10x4096x128xf16>, %v: tensor<2x10x4096x128xf16>)
                     -> tensor<2x10x4096x128xf16> {
  %scale = arith.constant 0.125 : f16
  %acc = tensor.empty() : tensor<2x10x4096x128xf16>
  %out = iree_linalg_ext.attention
         {indexing_maps = [#map, #map1, #map2, #map3, #map4]}
         ins(%q, %k, %v, %scale : tensor<2x10x4096x128xf16>, tensor<2x10x4096x128xf16>, tensor<2x10x4096x128xf16>, f16)
         outs(%acc : tensor<2x10x4096x128xf16>) -> tensor<2x10x4096x128xf16>
  func.return %out : tensor<2x10x4096x128xf16>
}

// CHECK-LABEL: func.func @attention
// CHECK-SAME: %[[Q:.+]]: tensor<2x10x4096x128xf16>, %[[K:.+]]: tensor<2x10x4096x128xf16>, %[[V:.+]]: tensor<2x10x4096x128xf16>
// CHECK-DAG: %[[ACC_INIT:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[MAX_INIT:.+]] = arith.constant -3.40282347E+38 : f32
// CHECK-DAG: %[[SUM_INIT:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[ACC_FILL:.+]] = linalg.fill ins(%[[ACC_INIT]]
// CHECK-DAG: %[[MAX_FILL:.+]] = linalg.fill ins(%[[MAX_INIT]]
// CHECK-DAG: %[[SUM_FILL:.+]] = linalg.fill ins(%[[SUM_INIT]]
// CHECK: %[[OUT:.+]]:3 = iree_linalg_ext.online_attention
// CHECK-SAME:         ins(%[[Q]], %[[K]], %[[V]]
// CHECK-SAME:         outs(%[[ACC_FILL]], %[[MAX_FILL]], %[[SUM_FILL]]
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[OUT]]#2, %[[OUT]]#0
// CHECK: arith.divf
// CHECK: arith.mulf
// CHECK: arith.truncf
// CHECK: linalg.yield
