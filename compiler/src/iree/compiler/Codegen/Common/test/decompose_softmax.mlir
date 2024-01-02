// RUN: iree-opt --split-input-file --iree-codegen-decompose-softmax -cse %s | FileCheck %s --check-prefix=CHECK-FUSE
// RUN: iree-opt --split-input-file --iree-codegen-decompose-softmax=use-fusion=false -cse %s | FileCheck %s --check-prefix=CHECK-FUSE
// RUN: iree-opt --split-input-file --iree-codegen-decompose-softmax=use-fusion=true -cse %s | FileCheck %s --check-prefix=CHECK-NO-FUSE

func.func @softmax(%arg0: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
  %0 = tensor.empty() : tensor<2x16x32xf32>
  %1 = linalg.softmax dimension(2) ins(%arg0 : tensor<2x16x32xf32>) outs(%0: tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
  return %1 : tensor<2x16x32xf32>
}

// CHECK-FUSE-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-FUSE-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-FUSE:      func.func @softmax(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
// CHECK-FUSE:        %[[D0:.+]] = tensor.empty() : tensor<2x16x32xf32>
// CHECK-FUSE:        %[[D1:.+]] = tensor.empty() : tensor<2x16xf32>
// CHECK-FUSE:        %[[CST:.+]] = arith.constant -3.40282347E+38 : f32
// CHECK-FUSE:        %[[D2:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D1]] : tensor<2x16xf32>) -> tensor<2x16xf32>
// CHECK-FUSE:        %[[D3:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]]], iterator_types = ["parallel",
// CHECK-FUSE-SAME:     "parallel", "reduction"]} ins(%[[ARG0]] : tensor<2x16x32xf32>) outs(%[[D2]] : tensor<2x16xf32>) {
// CHECK-FUSE:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK-FUSE:          %[[D8:.+]] = arith.maximumf %[[IN]], %[[OUT]] : f32
// CHECK-FUSE:          linalg.yield %[[D8]] : f32
// CHECK-FUSE:        } -> tensor<2x16xf32>
// CHECK-FUSE:        %[[D4:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP]]], iterator_types = ["parallel",
// CHECK-FUSE-SAME:     "parallel", "parallel"]} ins(%[[ARG0]], %3 : tensor<2x16x32xf32>, tensor<2x16xf32>) outs(%[[D0]] : tensor<2x16x32xf32>) {
// CHECK-FUSE:        ^bb0(%[[IN:.+]]: f32, %[[IN_1:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK-FUSE:          %[[D8:.+]] = arith.subf %[[IN]], %[[IN_1]] : f32
// CHECK-FUSE:          %[[D9:.+]] = math.exp %[[D8]] : f32
// CHECK-FUSE:          linalg.yield %[[D9]] : f32
// CHECK-FUSE:        } -> tensor<2x16x32xf32>
// CHECK-FUSE:        %[[CST0:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-FUSE:        %[[D5:.+]] = linalg.fill ins(%[[CST0]] : f32) outs(%[[D1]] : tensor<2x16xf32>) -> tensor<2x16xf32>
// CHECK-FUSE:        %[[D6:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]]], iterator_types =
// CHECK-FUSE-SAME:     ["parallel", "parallel", "reduction"]} ins(%[[D4]] : tensor<2x16x32xf32>)
// CHECK-FUSE-SAME:     outs(%[[D5]] : tensor<2x16xf32>) {
// CHECK-FUSE:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK-FUSE:          %[[D8:.+]] = arith.addf %[[IN]], %[[OUT]]
// CHECK-FUSE:          linalg.yield %[[D8]] : f32
// CHECK-FUSE:        } -> tensor<2x16xf32>
// CHECK-FUSE:        %[[D7:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP]]], iterator_types =
// CHECK-FUSE-SAME:     ["parallel", "parallel", "parallel"]} ins(%[[D4]], %[[D6]] : tensor<2x16x32xf32>, tensor<2x16xf32>)
// CHECK-FUSE-SAME:     outs(%[[D0]] : tensor<2x16x32xf32>) {
// CHECK-FUSE:        ^bb0(%[[IN:.+]]: f32, %[[IN_1:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK-FUSE:          %[[D8:.+]] = arith.divf %[[IN]], %[[IN_1]] : f32
// CHECK-FUSE:          linalg.yield %[[D8]] : f32
// CHECK-FUSE:        } -> tensor<2x16x32xf32>
// CHECK-FUSE:        return %[[D7]] : tensor<2x16x32xf32>
// CHECK-FUSE:      }

// CHECK-NO-FUSE-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-NO-FUSE-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-NO-FUSE:      func.func @softmax(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
// CHECK-NO-FUSE:        %[[D0:.+]] = tensor.empty() : tensor<2x16x32xf32>
// CHECK-NO-FUSE:        %[[D1:.+]] = tensor.empty() : tensor<2x16xf32>
// CHECK-NO-FUSE:        %[[CST:.+]] = arith.constant -3.40282347E+38 : f32
// CHECK-NO-FUSE:        %[[D2:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D1]] : tensor<2x16xf32>) -> tensor<2x16xf32>
// CHECK-NO-FUSE:        %[[D3:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]]], iterator_types = ["parallel",
// CHECK-NO-FUSE-SAME:     "parallel", "reduction"]} ins(%[[ARG0]] : tensor<2x16x32xf32>) outs(%[[D2]] : tensor<2x16xf32>) {
// CHECK-NO-FUSE:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK-NO-FUSE:          %[[D8:.+]] = arith.maximumf %[[IN]], %[[OUT]] : f32
// CHECK-NO-FUSE:          linalg.yield %[[D8]] : f32
// CHECK-NO-FUSE:        } -> tensor<2x16xf32>
// CHECK-NO-FUSE:        %[[CST0:.+]] = arith.constant 0.0
// CHECK-NO-FUSE:        %[[D4:.+]] = linalg.fill ins(%[[CST0]] : f32) outs(%[[D1]] : tensor<2x16xf32>) -> tensor<2x16xf32>
// CHECK-NO-FUSE:        %[[D5:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP1]]], iterator_types =
// CHECK-NO-FUSE-SAME:     ["parallel", "parallel", "reduction"]} ins(%[[ARG0]], %[[D3]] : tensor<2x16x32xf32>, tensor<2x16xf32>)
// CHECK-NO-FUSE-SAME:     outs(%[[D4]] : tensor<2x16xf32>) {
// CHECK-NO-FUSE:        ^bb0(%[[IN:.+]]: f32, %[[IN_1:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK-NO-FUSE:          %[[D8:.+]] = arith.subf %[[IN]], %[[IN_1]] : f32
// CHECK-NO-FUSE:          %[[D9:.+]] = math.exp %[[D8]] : f32
// CHECK-NO-FUSE:          %[[D10:.+]] = arith.addf %[[D9]], %[[OUT]]
// CHECK-NO-FUSE:          linalg.yield %[[D10]] : f32
// CHECK-NO-FUSE:        } -> tensor<2x16xf32>
// CHECK-NO-FUSE:        %[[D7:.+]]:2 = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP1]], #[[MAP]], #[[MAP]]], iterator_types =
// CHECK-NO-FUSE-SAME:     ["parallel", "parallel", "parallel"]} ins(%[[ARG0]], %[[D3]], %[[D5]] : tensor<2x16x32xf32>, tensor<2x16xf32>, tensor<2x16xf32>)
// CHECK-NO-FUSE-SAME:     outs(%[[D0]], %[[D0]] : tensor<2x16x32xf32>, tensor<2x16x32xf32>) {
// CHECK-NO-FUSE:        ^bb0(%[[IN:.+]]: f32, %[[IN_1:.+]]: f32, %[[IN_2:.+]]: f32, %[[OUT0:.+]]: f32, %[[OUT1:.+]]: f32):
// CHECK-NO-FUSE:          %[[D8:.+]] = arith.subf %[[IN]], %[[IN_1]] : f32
// CHECK-NO-FUSE:          %[[D9:.+]] = math.exp %[[D8]] : f32
// CHECK-NO-FUSE:          %[[D10:.+]] = arith.divf %[[D9]], %[[IN_2]] : f32
// CHECK-NO-FUSE:          linalg.yield %[[D9]], %[[D10]] : f32
// CHECK-NO-FUSE:        } -> (tensor<2x16x32xf32>, tensor<2x16x32xf32>)
// CHECK-NO-FUSE:        return %[[D7]]#1 : tensor<2x16x32xf32>
// CHECK-NO-FUSE:      }
