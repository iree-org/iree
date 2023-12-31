// RUN: iree-opt --split-input-file --iree-codegen-decompose-softmax=use-fusion=true -cse %s | FileCheck %s

func.func @softmax(%arg0: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
  %0 = tensor.empty() : tensor<2x16x32xf32>
  %1 = linalg.softmax dimension(2) ins(%arg0 : tensor<2x16x32xf32>) outs(%0: tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
  return %1 : tensor<2x16x32xf32>
}

// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK:      func.func @softmax(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<2x16x32xf32>
// CHECK:        %[[D1:.+]] = tensor.empty() : tensor<2x16xf32>
// CHECK:        %[[CST:.+]] = arith.constant -3.40282347E+38 : f32
// CHECK:        %[[D2:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D1]] : tensor<2x16xf32>) -> tensor<2x16xf32>
// CHECK:        %[[D3:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]]], iterator_types = ["parallel",
// CHECK-SAME:     "parallel", "reduction"]} ins(%[[ARG0]] : tensor<2x16x32xf32>) outs(%[[D2]] : tensor<2x16xf32>) {
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:          %[[D8:.+]] = arith.maximumf %[[IN]], %[[OUT]] : f32
// CHECK:          linalg.yield %[[D8]] : f32
// CHECK:        } -> tensor<2x16xf32>
// CHECK:        %[[CST0:.+]] = arith.constant 0.0
// CHECK:        %[[D4:.+]] = linalg.fill ins(%[[CST0]] : f32) outs(%[[D1]] : tensor<2x16xf32>) -> tensor<2x16xf32>
// CHECK:        %[[D5:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP1]]], iterator_types =
// CHECK-SAME:     ["parallel", "parallel", "reduction"]} ins(%[[ARG0]], %[[D3]] : tensor<2x16x32xf32>, tensor<2x16xf32>)
// CHECK-SAME:     outs(%[[D4]] : tensor<2x16xf32>) {
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[IN_1:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:          %[[D8:.+]] = arith.subf %[[IN]], %[[IN_1]] : f32
// CHECK:          %[[D9:.+]] = math.exp %[[D8]] : f32
// CHECK:          %[[D10:.+]] = arith.addf %[[D9]], %[[OUT]]
// CHECK:          linalg.yield %[[D10]] : f32
// CHECK:        } -> tensor<2x16xf32>
// CHECK:        %[[D7:.+]]:2 = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP1]], #[[MAP]], #[[MAP]]], iterator_types =
// CHECK-SAME:     ["parallel", "parallel", "parallel"]} ins(%[[ARG0]], %[[D3]], %[[D5]] : tensor<2x16x32xf32>, tensor<2x16xf32>, tensor<2x16xf32>)
// CHECK-SAME:     outs(%[[D0]], %[[D0]] : tensor<2x16x32xf32>, tensor<2x16x32xf32>) {
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[IN_1:.+]]: f32, %[[IN_2:.+]]: f32, %[[OUT0:.+]]: f32, %[[OUT1:.+]]: f32):
// CHECK:          %[[D8:.+]] = arith.subf %[[IN]], %[[IN_1]] : f32
// CHECK:          %[[D9:.+]] = math.exp %[[D8]] : f32
// CHECK:          %[[D10:.+]] = arith.divf %[[D9]], %[[IN_2]] : f32
// CHECK:          linalg.yield %[[D9]], %[[D10]] : f32
// CHECK:        } -> (tensor<2x16x32xf32>, tensor<2x16x32xf32>)
// CHECK:        return %[[D7]]#1 : tensor<2x16x32xf32>
// CHECK:      }
