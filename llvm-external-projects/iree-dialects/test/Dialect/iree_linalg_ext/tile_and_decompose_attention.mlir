// RUN: iree-dialects-opt --split-input-file -iree-linalg-ext-tile-and-decompose-attention -cse %s | FileCheck %s

func.func @attention(%query: tensor<192x1024x64xf32>, %key: tensor<192x1024x64xf32>, %value: tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32> {
  %0 = tensor.empty() : tensor<192x1024x64xf32>
  %1 = iree_linalg_ext.attention ins(%query, %key, %value : tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, tensor<192x1024x64xf32>) outs(%0 : tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
  return %1 : tensor<192x1024x64xf32>
}

// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0)>
// CHECK:      func.func @attention(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<192x1024x64xf32>, %[[ARG1:[a-zA-Z0-9_]+]]:
// CHECK-SAME:   tensor<192x1024x64xf32>, %[[ARG2:[a-zA-Z0-9_]+]]: tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
// CHECK-SAME:   {
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<192x1024x64xf32>
// CHECK-DAG:      %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:      %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:      %[[C192:.+]] = arith.constant 192 : index
// CHECK:        %[[D1:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C192]] step %[[C1]]
// CHECK-SAME:     iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[D0]]) -> (tensor<192x1024x64xf32>) {
// CHECK-DAG:        %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:        %[[CST_0:.+]] = arith.constant -1.000000e+30 : f32
// CHECK:          %[[D2:.+]] = tensor.empty() : tensor<1024xf32>
// CHECK:          %[[D3:.+]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[D2]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:          %[[D4:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D2]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-DAG:        %[[C1024:.+]] = arith.constant 1024 : index
// CHECK:          %[[D5:.+]]:3 = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1024]] step %[[C1024]]
// CHECK-SAME:       iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[ARG4]], %[[ARG7:[a-zA-Z0-9_]+]] = %[[D3]],
// CHECK-SAME:       %[[ARG8:[a-zA-Z0-9_]+]] = %[[D4]]) -> (tensor<192x1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>)
// CHECK-SAME:       {
// CHECK:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG3]], %[[ARG5]], 0] [1, 1024, 64] [1,
// CHECK-SAME:         1, 1] : tensor<192x1024x64xf32> to tensor<1024x64xf32>
// CHECK:            %[[EXTRACTED_SLICE_1:.+]] = tensor.extract_slice %[[ARG2]][%[[ARG3]], %[[ARG5]], 0] [1, 1024, 64]
// CHECK-SAME:         [1, 1, 1] : tensor<192x1024x64xf32> to tensor<1024x64xf32>
// CHECK:            %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG3]], 0, 0] [1, 1024, 64] [1, 1, 1]
// CHECK-SAME:         : tensor<192x1024x64xf32> to tensor<1024x64xf32>
// CHECK:            %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[ARG6]][%[[ARG3]], 0, 0] [1, 1024, 64] [1, 1, 1]
// CHECK-SAME:         : tensor<192x1024x64xf32> to tensor<1024x64xf32>
// CHECK:            %[[D6:.+]] = tensor.empty() : tensor<1024x1024xf32>
// CHECK:            %[[D7:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D6]] : tensor<1024x1024xf32>) ->
// CHECK-SAME:         tensor<1024x1024xf32>
// CHECK:            %[[D8:.+]] = linalg.matmul_transpose_b ins(%[[EXTRACTED_SLICE_2]], %[[EXTRACTED_SLICE]] :
// CHECK-SAME:         tensor<1024x64xf32>, tensor<1024x64xf32>) outs(%[[D7]] : tensor<1024x1024xf32>) ->
// CHECK-SAME:         tensor<1024x1024xf32>
// CHECK:            %[[D9:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]]], iterator_types = ["parallel",
// CHECK-SAME:         "reduction"]} ins(%[[D8]] : tensor<1024x1024xf32>) outs(%[[ARG7]] : tensor<1024xf32>) {
// CHECK:            ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:              %[[D17:.+]] = arith.maxf %[[IN]], %[[OUT]] : f32
// CHECK:              linalg.yield %[[D17]] : f32
// CHECK:            } -> tensor<1024xf32>
// CHECK:            %[[D10:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP]]], iterator_types = ["parallel",
// CHECK-SAME:         "parallel"]} ins(%[[D9]] : tensor<1024xf32>) outs(%[[D8]] : tensor<1024x1024xf32>) {
// CHECK:            ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:              %[[D17]] = arith.subf %[[OUT]], %[[IN]] : f32
// CHECK:              %[[D18:.+]] = math.exp %[[D17]] : f32
// CHECK:              linalg.yield %[[D18]] : f32
// CHECK:            } -> tensor<1024x1024xf32>
// CHECK:            %[[D11:.+]] = linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP2]], #[[MAP2]]], iterator_types =
// CHECK-SAME:         ["parallel"]} ins(%[[ARG7]], %[[D9]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[ARG8]] :
// CHECK-SAME:         tensor<1024xf32>) {
// CHECK:            ^bb0(%[[IN:.+]]: f32, %[[IN_4:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:              %[[D17]] = arith.subf %[[IN]], %[[IN_4]] : f32
// CHECK:              %[[D18]] = math.exp %[[D17]] : f32
// CHECK:              %[[D19:.+]] = arith.mulf %[[D18]], %[[OUT]] : f32
// CHECK:              linalg.yield %[[D19]] : f32
// CHECK:            } -> tensor<1024xf32>
// CHECK:            %[[D12:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]]], iterator_types = ["parallel",
// CHECK-SAME:         "reduction"]} ins(%[[D10]] : tensor<1024x1024xf32>) outs(%[[D11]] : tensor<1024xf32>) {
// CHECK:            ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:              %[[D17]] = arith.addf %[[IN]], %[[OUT]] : f32
// CHECK:              linalg.yield %[[D17]] : f32
// CHECK:            } -> tensor<1024xf32>
// CHECK:            %[[D13:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP]]], iterator_types = ["parallel",
// CHECK-SAME:         "parallel"]} ins(%[[D12]] : tensor<1024xf32>) outs(%[[D10]] : tensor<1024x1024xf32>) {
// CHECK:            ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:              %[[D17]] = arith.divf %[[OUT]], %[[IN]] : f32
// CHECK:              linalg.yield %[[D17]] : f32
// CHECK:            } -> tensor<1024x1024xf32>
// CHECK:            %[[D14:.+]] = tensor.empty() : tensor<1024x64xf32>
// CHECK:            %[[D15:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP1]], #[[MAP]]],
// CHECK-SAME:         iterator_types = ["parallel", "parallel"]} ins(%[[EXTRACTED_SLICE_3]], %[[D11]], %[[D12]] :
// CHECK-SAME:         tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>) outs(%[[D14]] : tensor<1024x64xf32>) {
// CHECK:            ^bb0(%[[IN:.+]]: f32, %[[IN_4:.+]]: f32, %[[IN_5:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:              %[[D17]] = arith.divf %[[IN_4]], %[[IN_5]] : f32
// CHECK:              %[[D18]] = arith.mulf %[[D17]], %[[IN]] : f32
// CHECK:              linalg.yield %[[D18]] : f32
// CHECK:            } -> tensor<1024x64xf32>
// CHECK:            %[[D16:.+]] = linalg.matmul ins(%[[D13]], %[[EXTRACTED_SLICE_1]] : tensor<1024x1024xf32>,
// CHECK-SAME:         tensor<1024x64xf32>) outs(%[[D15]] : tensor<1024x64xf32>) -> tensor<1024x64xf32>
// CHECK:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D16]] into %[[ARG6]][%[[ARG3]], 0, 0] [1, 1024, 64]
// CHECK-SAME:         [1, 1, 1] : tensor<1024x64xf32> into tensor<192x1024x64xf32>
// CHECK:            scf.yield %[[INSERTED_SLICE]], %[[D9]], %[[D12]] : tensor<192x1024x64xf32>, tensor<1024xf32>,
// CHECK-SAME:         tensor<1024xf32>
// CHECK:          }
// CHECK:          scf.yield %[[D5]]#[[D0:.+]] : tensor<192x1024x64xf32>
// CHECK:        }
// CHECK:        return %[[D1]] : tensor<192x1024x64xf32>
// CHECK:      }
