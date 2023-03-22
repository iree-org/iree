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
// CHECK:          %[[D2:.+]] = tensor.empty() : tensor<1x1024xf32>
// CHECK:          %[[D3:.+]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[D2]] : tensor<1x1024xf32>) ->
// CHECK-SAME:       tensor<1x1024xf32>
// CHECK:          %[[D4:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D2]] : tensor<1x1024xf32>) ->
// CHECK-SAME:       tensor<1x1024xf32>
// CHECK-DAG:        %[[C1024:.+]] = arith.constant 1024 : index
// CHECK:          %[[D5:.+]]:3 = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1024]] step %[[C1024]]
// CHECK-SAME:       iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[ARG4]], %[[ARG7:[a-zA-Z0-9_]+]] = %[[D3]],
// CHECK-SAME:       %[[ARG8:[a-zA-Z0-9_]+]] = %[[D4]]) -> (tensor<192x1024x64xf32>, tensor<1x1024xf32>,
// CHECK-SAME:       tensor<1x1024xf32>) {
// CHECK:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG3]], %[[ARG5]], 0] [1, 1024, 64] [1,
// CHECK-SAME:         1, 1] : tensor<192x1024x64xf32> to tensor<1024x64xf32>
// CHECK:            %[[EXTRACTED_SLICE_1:.+]] = tensor.extract_slice %[[ARG2]][%[[ARG3]], %[[ARG5]], 0] [1, 1024, 64]
// CHECK-SAME:         [1, 1, 1] : tensor<192x1024x64xf32> to tensor<1024x64xf32>
// CHECK-DAG:          %[[C32:.+]] = arith.constant 32 : index
// CHECK:            %[[D6:.+]]:3 = scf.for %[[ARG9:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1024]] step %[[C32]]
// CHECK-SAME:         iter_args(%[[ARG10:[a-zA-Z0-9_]+]] = %[[ARG6]], %[[ARG11:[a-zA-Z0-9_]+]] = %[[ARG7]],
// CHECK-SAME:         %[[ARG12:[a-zA-Z0-9_]+]] = %[[ARG8]]) -> (tensor<192x1024x64xf32>, tensor<1x1024xf32>,
// CHECK-SAME:         tensor<1x1024xf32>) {
// CHECK:              %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[ARG0]][0, %[[ARG9]], 0] [1, 32, 64] [1, 1, 1]
// CHECK-SAME:           : tensor<192x1024x64xf32> to tensor<32x64xf32>
// CHECK:              %[[EXTRACTED_SLICE_4:.+]] = tensor.extract_slice %[[ARG10]][0, %[[ARG9]], 0] [1, 32, 64] [1, 1,
// CHECK-SAME:           1] : tensor<192x1024x64xf32> to tensor<32x64xf32>
// CHECK:              %[[EXTRACTED_SLICE_5:.+]] = tensor.extract_slice %[[ARG11]][0, %[[ARG9]]] [1, 32] [1, 1] :
// CHECK-SAME:           tensor<1x1024xf32> to tensor<32xf32>
// CHECK:              %[[EXTRACTED_SLICE_6:.+]] = tensor.extract_slice %[[ARG12]][0, %[[ARG9]]] [1, 32] [1, 1] :
// CHECK-SAME:           tensor<1x1024xf32> to tensor<32xf32>
// CHECK:              %[[D7:.+]] = tensor.empty() : tensor<32x1024xf32>
// CHECK:              %[[D8:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D7]] : tensor<32x1024xf32>) ->
// CHECK-SAME:           tensor<32x1024xf32>
// CHECK:              %[[D9:.+]] = linalg.matmul_transpose_b ins(%[[EXTRACTED_SLICE_3]], %[[EXTRACTED_SLICE]] :
// CHECK-SAME:           tensor<32x64xf32>, tensor<1024x64xf32>) outs(%[[D8]] : tensor<32x1024xf32>) ->
// CHECK-SAME:           tensor<32x1024xf32>
// CHECK:              %[[D10:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]]], iterator_types =
// CHECK-SAME:           ["parallel", "reduction"]} ins(%[[D9]] : tensor<32x1024xf32>) outs(%[[EXTRACTED_SLICE_5]] :
// CHECK-SAME:           tensor<32xf32>) {
// CHECK:              ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:                %[[D18:.+]] = arith.maxf %[[IN]], %[[OUT]] : f32
// CHECK:                linalg.yield %[[D18]] : f32
// CHECK:              } -> tensor<32xf32>
// CHECK:              %[[D11:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP]]], iterator_types =
// CHECK-SAME:           ["parallel", "parallel"]} ins(%[[D10]] : tensor<32xf32>) outs(%[[D9]] : tensor<32x1024xf32>) {
// CHECK:              ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:                %[[D18]] = arith.subf %[[OUT]], %[[IN]] : f32
// CHECK:                %[[D19:.+]] = math.exp %[[D18]] : f32
// CHECK:                linalg.yield %[[D19]] : f32
// CHECK:              } -> tensor<32x1024xf32>
// CHECK:              %[[D12:.+]] = linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP2]], #[[MAP2]]], iterator_types =
// CHECK-SAME:           ["parallel"]} ins(%[[EXTRACTED_SLICE_5]], %[[D10]] : tensor<32xf32>, tensor<32xf32>)
// CHECK-SAME:           outs(%[[EXTRACTED_SLICE_6]] : tensor<32xf32>) {
// CHECK:              ^bb0(%[[IN:.+]]: f32, %[[IN_10:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:                %[[D18]] = arith.subf %[[IN]], %[[IN_10]] : f32
// CHECK:                %[[D19]] = math.exp %[[D18]] : f32
// CHECK:                %[[D20:.+]] = arith.mulf %[[D19]], %[[OUT]] : f32
// CHECK:                linalg.yield %[[D20]] : f32
// CHECK:              } -> tensor<32xf32>
// CHECK:              %[[D13:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]]], iterator_types =
// CHECK-SAME:           ["parallel", "reduction"]} ins(%[[D11]] : tensor<32x1024xf32>) outs(%[[D12]] : tensor<32xf32>)
// CHECK-SAME:           {
// CHECK:              ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:                %[[D18]] = arith.addf %[[IN]], %[[OUT]] : f32
// CHECK:                linalg.yield %[[D18]] : f32
// CHECK:              } -> tensor<32xf32>
// CHECK:              %[[D14:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP]]], iterator_types =
// CHECK-SAME:           ["parallel", "parallel"]} ins(%[[D13]] : tensor<32xf32>) outs(%[[D11]] : tensor<32x1024xf32>)
// CHECK-SAME:           {
// CHECK:              ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:                %[[D18]] = arith.divf %[[OUT]], %[[IN]] : f32
// CHECK:                linalg.yield %[[D18]] : f32
// CHECK:              } -> tensor<32x1024xf32>
// CHECK:              %[[D15:.+]] = tensor.empty() : tensor<32x64xf32>
// CHECK:              %[[D16:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP1]], #[[MAP]]],
// CHECK-SAME:           iterator_types = ["parallel", "parallel"]} ins(%[[EXTRACTED_SLICE_4]], %[[D12]], %[[D13]] :
// CHECK-SAME:           tensor<32x64xf32>, tensor<32xf32>, tensor<32xf32>) outs(%[[D15]] : tensor<32x64xf32>) {
// CHECK:              ^bb0(%[[IN:.+]]: f32, %[[IN_10:.+]]: f32, %[[IN_11:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:                %[[D18]] = arith.divf %[[IN_10]], %[[IN_11]] : f32
// CHECK:                %[[D19]] = arith.mulf %[[D18]], %[[IN]] : f32
// CHECK:                linalg.yield %[[D19]] : f32
// CHECK:              } -> tensor<32x64xf32>
// CHECK:              %[[D17:.+]] = linalg.matmul ins(%[[D14]], %[[EXTRACTED_SLICE_1]] : tensor<32x1024xf32>,
// CHECK-SAME:           tensor<1024x64xf32>) outs(%[[D16]] : tensor<32x64xf32>) -> tensor<32x64xf32>
// CHECK:              %[[INSERTED_SLICE_7:.+]] = tensor.insert_slice %[[D17]] into %[[ARG10]][%[[ARG9]], 0, 0] [1, 32,
// CHECK-SAME:           64] [1, 1, 1] : tensor<32x64xf32> into tensor<192x1024x64xf32>
// CHECK:              %[[INSERTED_SLICE_8:.+]] = tensor.insert_slice %[[D10]] into %[[ARG11]][0, %[[ARG9]]] [1, 32] [1,
// CHECK-SAME:           1] : tensor<32xf32> into tensor<1x1024xf32>
// CHECK:              %[[INSERTED_SLICE_9:.+]] = tensor.insert_slice %[[D13]] into %[[ARG12]][0, %[[ARG9]]] [1, 32] [1,
// CHECK-SAME:           1] : tensor<32xf32> into tensor<1x1024xf32>
// CHECK:              scf.yield %[[INSERTED_SLICE_7]], %[[INSERTED_SLICE_8]], %[[INSERTED_SLICE_9]] :
// CHECK-SAME:           tensor<192x1024x64xf32>, tensor<1x1024xf32>, tensor<1x1024xf32>
// CHECK:            }
// CHECK:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D6]]#[[D1:.+]] into %[[ARG7]][0, 0] [1, 1024] [1,
// CHECK-SAME:         1] : tensor<1x1024xf32> into tensor<1x1024xf32>
// CHECK:            %[[INSERTED_SLICE_2:.+]] = tensor.insert_slice %[[D6]]#[[D2:.+]] into %[[ARG8]][0, 0] [1, 1024] [1,
// CHECK-SAME:         1] : tensor<1x1024xf32> into tensor<1x1024xf32>
// CHECK:            scf.yield %[[D6]]#[[D0:.+]], %[[INSERTED_SLICE]], %[[INSERTED_SLICE_2]] : tensor<192x1024x64xf32>,
// CHECK-SAME:         tensor<1x1024xf32>, tensor<1x1024xf32>
// CHECK:          }
// CHECK:          scf.yield %[[D5]]#[[D0]] : tensor<192x1024x64xf32>
// CHECK:        }
// CHECK:        return %[[D1]] : tensor<192x1024x64xf32>
// CHECK:      }
