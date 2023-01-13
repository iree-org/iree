// RUN: iree-dialects-opt --iree-linalg-ext-tile-and-decompose-flash-attention --split-input-file %s | FileCheck %s

#map = affine_map<(d0)[s0, s1] -> (2, -d0 + s1)>
#map1 = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
module {
  func.func @flash_attention_fwd(%arg0: tensor<192x1024x64xf32>, %arg1: tensor<192x1024x64xf32>, %arg2: tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32> {
    %c0 = arith.constant 0 : index
    %c192 = arith.constant 192 : index
    %c1024 = arith.constant 1024 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %0 = tensor.empty() : tensor<192x1024x64xf32>
    %1 = scf.for %arg3 = %c0 to %c192 step %c2 iter_args(%arg4 = %0) -> (tensor<192x1024x64xf32>) {
      %2 = affine.min #map(%arg3)[%c2, %c192]
      %3 = scf.for %arg5 = %c0 to %c1024 step %c32 iter_args(%arg6 = %arg4) -> (tensor<192x1024x64xf32>) {
        %4 = affine.min #map1(%arg5)[%c32, %c1024]
        %extracted_slice = tensor.extract_slice %arg0[%arg3, %arg5, 0] [%2, %4, 64] [1, 1, 1] : tensor<192x1024x64xf32> to tensor<?x?x64xf32>
        %extracted_slice_0 = tensor.extract_slice %arg1[%arg3, 0, 0] [%2, 1024, 64] [1, 1, 1] : tensor<192x1024x64xf32> to tensor<?x1024x64xf32>
        %extracted_slice_1 = tensor.extract_slice %arg2[%arg3, 0, 0] [%2, 1024, 64] [1, 1, 1] : tensor<192x1024x64xf32> to tensor<?x1024x64xf32>
        %extracted_slice_2 = tensor.extract_slice %0[%arg3, %arg5, 0] [%2, %4, 64] [1, 1, 1] : tensor<192x1024x64xf32> to tensor<?x?x64xf32>
        %5 = iree_linalg_ext.flash_attention.fwd ins(%extracted_slice, %extracted_slice_0, %extracted_slice_1 : tensor<?x?x64xf32>, tensor<?x1024x64xf32>, tensor<?x1024x64xf32>) outs(%extracted_slice_2 : tensor<?x?x64xf32>) -> tensor<?x?x64xf32>
        %inserted_slice = tensor.insert_slice %5 into %arg6[%2, %4, 0] [%arg3, %arg5, 64] [1, 1, 1] : tensor<?x?x64xf32> into tensor<192x1024x64xf32>
        scf.yield %inserted_slice : tensor<192x1024x64xf32>
      }
      scf.yield %3 : tensor<192x1024x64xf32>
    }
    return %1 : tensor<192x1024x64xf32>
  }
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (2, -d0 + s1)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG:  #[[MAP4:.+]] = affine_map<(d0) -> (d0)>
// CHECK:      func.func @flash_attention_fwd(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<192x1024x64xf32>, %[[ARG1:[a-zA-Z0-9_]+]]:
// CHECK-SAME:   tensor<192x1024x64xf32>, %[[ARG2:[a-zA-Z0-9_]+]]: tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
// CHECK-SAME:   {
// CHECK:        %[[C1:.+]] = arith.constant 1 : index
// CHECK:        %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[CST_0:.+]] = arith.constant -1.000000e+02 : f32
// CHECK:        %[[C0:.+]] = arith.constant 0 : index
// CHECK:        %[[C192:.+]] = arith.constant 192 : index
// CHECK:        %[[C1024:.+]] = arith.constant 1024 : index
// CHECK:        %[[C2:.+]] = arith.constant 2 : index
// CHECK:        %[[C32:.+]] = arith.constant 32 : index
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<192x1024x64xf32>
// CHECK:        %[[D1:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C192]] step %[[C2]]
// CHECK-SAME:     iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[D0]]) -> (tensor<192x1024x64xf32>) {
// CHECK-DAG:        %[[D2:.+]] = affine.min #[[MAP]](%[[ARG3]])[%[[C2]], %[[C192]]]
// CHECK:          %[[D3:.+]] = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1024]] step %[[C32]]
// CHECK-SAME:       iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[ARG4]]) -> (tensor<192x1024x64xf32>) {
// CHECK-DAG:          %[[D4:.+]] = affine.min #[[MAP1]](%[[ARG5]])[%[[C32]], %[[C1024]]]
// CHECK:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG3]], %[[ARG5]], 0] [%[[D2]],
// CHECK-SAME:         %[[D4]], 64] [1, 1, 1] : tensor<192x1024x64xf32> to tensor<?x?x64xf32>
// CHECK:            %[[EXTRACTED_SLICE_1:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG3]], 0, 0] [%[[D2]], 1024, 64] [1,
// CHECK-SAME:         1, 1] : tensor<192x1024x64xf32> to tensor<?x1024x64xf32>
// CHECK:            %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[ARG2]][%[[ARG3]], 0, 0] [%[[D2]], 1024, 64] [1,
// CHECK-SAME:         1, 1] : tensor<192x1024x64xf32> to tensor<?x1024x64xf32>
// CHECK:            %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[D0]][%[[ARG3]], %[[ARG5]], 0] [%[[D2]],
// CHECK-SAME:         %[[D4]], 64] [1, 1, 1] : tensor<192x1024x64xf32> to tensor<?x?x64xf32>
// CHECK:            %[[D5:.+]] = scf.for %[[ARG7:[a-zA-Z0-9_]+]] = %[[C0]] to %[[D2]] step %[[C1]]
// CHECK-SAME:         iter_args(%[[ARG8:[a-zA-Z0-9_]+]] = %[[EXTRACTED_SLICE_3]]) -> (tensor<?x?x64xf32>) {
// CHECK:              %[[D6:.+]] = tensor.empty(%[[D4]]) : tensor<?xf32>
// CHECK:              %[[D7:.+]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[D6]] : tensor<?xf32>) -> tensor<?xf32>
// CHECK:              %[[D8:.+]] = tensor.empty(%[[D4]]) : tensor<?xf32>
// CHECK:              %[[D9:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D8]] : tensor<?xf32>) -> tensor<?xf32>
// CHECK:              %[[D10:.+]]:3 = scf.for %[[ARG9:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1024]] step %[[D4]]
// CHECK-SAME:           iter_args(%[[ARG10:[a-zA-Z0-9_]+]] = %[[ARG8]], %[[ARG11:[a-zA-Z0-9_]+]] = %[[D7]],
// CHECK-SAME:           %[[ARG12:[a-zA-Z0-9_]+]] = %[[D9]]) -> (tensor<?x?x64xf32>, tensor<?xf32>, tensor<?xf32>) {
// CHECK:                %[[EXTRACTED_SLICE_4:.+]] = tensor.extract_slice %[[EXTRACTED_SLICE_1]][%[[ARG7]], %[[ARG9]],
// CHECK-SAME:             0] [1, %[[D4]], 64] [1, 1, 1] : tensor<?x1024x64xf32> to tensor<?x64xf32>
// CHECK:                %[[EXTRACTED_SLICE_5:.+]] = tensor.extract_slice %[[EXTRACTED_SLICE_2]][%[[ARG7]], %[[ARG9]],
// CHECK-SAME:             0] [1, %[[D4]], 64] [1, 1, 1] : tensor<?x1024x64xf32> to tensor<?x64xf32>
// CHECK:                %[[EXTRACTED_SLICE_6:.+]] = tensor.extract_slice %[[EXTRACTED_SLICE]][%[[ARG7]], 0, 0] [1,
// CHECK-SAME:             %[[D4]], 64] [1, 1, 1] : tensor<?x?x64xf32> to tensor<?x64xf32>
// CHECK:                %[[EXTRACTED_SLICE_7:.+]] = tensor.extract_slice %[[ARG10]][%[[ARG7]], 0, 0] [1, %[[D4]], 64]
// CHECK-SAME:             [1, 1, 1] : tensor<?x?x64xf32> to tensor<?x64xf32>
// CHECK:                %[[D11:.+]] = tensor.empty(%[[D4]]) : tensor<64x?xf32>
// CHECK:                %[[D12:.+]] = tensor.empty(%[[D4]], %[[D4]]) : tensor<?x?xf32>
// CHECK:                %[[TRANSPOSED:.+]] = linalg.transpose ins(%[[EXTRACTED_SLICE_4]] : tensor<?x64xf32>)
// CHECK-SAME:             outs(%[[D11]] : tensor<64x?xf32>) permutation = [1, 0]
// CHECK:                %[[D13:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D12]] : tensor<?x?xf32>) ->
// CHECK-SAME:             tensor<?x?xf32>
// CHECK:                %[[D14:.+]] = linalg.matmul ins(%[[EXTRACTED_SLICE_6]], %[[TRANSPOSED]] : tensor<?x64xf32>,
// CHECK-SAME:             tensor<64x?xf32>) outs(%[[D13]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:                %[[D15:.+]] = tensor.empty(%[[D4]]) : tensor<?xf32>
// CHECK:                %[[D16:.+]] = linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP3]]], iterator_types =
// CHECK-SAME:             ["reduction", "parallel"]} ins(%[[D14]] : tensor<?x?xf32>) outs(%[[D15]] : tensor<?xf32>) {
// CHECK:                ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:                  %[[D26:.+]] = arith.maxf %[[IN]], %[[OUT]] : f32
// CHECK:                  linalg.yield %[[D26]] : f32
// CHECK:                } -> tensor<?xf32>
// CHECK:                %[[D17:.+]] = linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP3]]], iterator_types =
// CHECK-SAME:             ["reduction", "parallel"]} ins(%[[D14]] : tensor<?x?xf32>) outs(%[[D15]] : tensor<?xf32>) {
// CHECK:                ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:                  %[[D26]] = arith.addf %[[IN]], %[[OUT]] : f32
// CHECK:                  linalg.yield %[[D26]] : f32
// CHECK:                } -> tensor<?xf32>
// CHECK:                %[[D18:.+]] = linalg.generic {indexing_maps = [#[[MAP4]], #[[MAP4]], #[[MAP4]]], iterator_types
// CHECK-SAME:             = ["parallel"]} ins(%[[ARG11]], %[[D16]] : tensor<?xf32>, tensor<?xf32>) outs(%[[D15]] :
// CHECK-SAME:             tensor<?xf32>) {
// CHECK:                ^bb0(%[[IN:.+]]: f32, %[[IN_11:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:                  %[[D26]] = arith.maxf %[[IN]], %[[IN_11]] : f32
// CHECK:                  linalg.yield %[[D26]] : f32
// CHECK:                } -> tensor<?xf32>
// CHECK:                %[[D19:.+]] = linalg.generic {indexing_maps = [#[[MAP4]], #[[MAP4]], #[[MAP4]]], iterator_types
// CHECK-SAME:             = ["parallel"]} ins(%[[ARG11]], %[[D18]] : tensor<?xf32>, tensor<?xf32>) outs(%[[D15]] :
// CHECK-SAME:             tensor<?xf32>) {
// CHECK:                ^bb0(%[[IN:.+]]: f32, %[[IN_11:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:                  %[[D26]] = arith.subf %[[IN]], %[[IN_11]] : f32
// CHECK:                  %[[D27:.+]] = math.exp %[[D26]] : f32
// CHECK:                  linalg.yield %[[D27]] : f32
// CHECK:                } -> tensor<?xf32>
// CHECK:                %[[D20:.+]] = linalg.generic {indexing_maps = [#[[MAP4]], #[[MAP4]], #[[MAP4]]], iterator_types
// CHECK-SAME:             = ["parallel"]} ins(%[[D16]], %[[D18]] : tensor<?xf32>, tensor<?xf32>) outs(%[[D15]] :
// CHECK-SAME:             tensor<?xf32>) {
// CHECK:                ^bb0(%[[IN:.+]]: f32, %[[IN_11:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:                  %[[D26]] = arith.subf %[[IN]], %[[IN_11]] : f32
// CHECK:                  %[[D27]] = math.exp %[[D26]] : f32
// CHECK:                  linalg.yield %[[D27]] : f32
// CHECK:                } -> tensor<?xf32>
// CHECK:                %[[D21:.+]] = linalg.generic {indexing_maps = [#[[MAP4]], #[[MAP4]], #[[MAP4]], #[[MAP4]],
// CHECK-SAME:             #[[MAP4]]], iterator_types = ["parallel"]} ins(%[[ARG12]], %[[D17]], %[[D19]], %[[D20]] :
// CHECK-SAME:             tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%[[D15]] : tensor<?xf32>) {
// CHECK:                ^bb0(%[[IN:.+]]: f32, %[[IN_11:.+]]: f32, %[[IN_12:.+]]: f32, %[[IN_13:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:                  %[[D26]] = arith.mulf %[[IN_12]], %[[IN]] : f32
// CHECK:                  %[[D27]] = arith.mulf %[[IN_13]], %[[IN_11]] : f32
// CHECK:                  %[[D28:.+]] = arith.addf %[[D26]], %[[D27]] : f32
// CHECK:                  linalg.yield %[[D28]] : f32
// CHECK:                } -> tensor<?xf32>
// CHECK:                %[[D22:.+]] = linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP3]], #[[MAP3]], #[[MAP3]],
// CHECK-SAME:             #[[MAP2]]], iterator_types = ["parallel", "parallel"]} ins(%[[D14]], %[[D16]], %[[D20]],
// CHECK-SAME:             %[[D21]] : tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%[[D12]] :
// CHECK-SAME:             tensor<?x?xf32>) {
// CHECK:                ^bb0(%[[IN:.+]]: f32, %[[IN_11:.+]]: f32, %[[IN_12:.+]]: f32, %[[IN_13:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:                  %[[D26]] = arith.subf %[[IN]], %[[IN_11]] : f32
// CHECK:                  %[[D27]] = math.exp %[[D26]] : f32
// CHECK:                  %[[D28]] = arith.mulf %[[D27]], %[[IN_12]] : f32
// CHECK:                  %[[D29:.+]] = arith.divf %[[D28]], %[[IN_13]] : f32
// CHECK:                  linalg.yield %[[D29]] : f32
// CHECK:                } -> tensor<?x?xf32>
// CHECK:                %[[D23:.+]] = tensor.empty(%[[C1024]]) : tensor<?x64xf32>
// CHECK:                %[[D24:.+]] = linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP3]], #[[MAP3]], #[[MAP3]],
// CHECK-SAME:             #[[MAP2]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXTRACTED_SLICE_7]],
// CHECK-SAME:             %[[ARG12]], %[[D21]], %[[D19]] : tensor<?x64xf32>, tensor<?xf32>, tensor<?xf32>,
// CHECK-SAME:             tensor<?xf32>) outs(%[[D23]] : tensor<?x64xf32>) {
// CHECK:                ^bb0(%[[IN:.+]]: f32, %[[IN_11:.+]]: f32, %[[IN_12:.+]]: f32, %[[IN_13:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:                  %[[D26]] = arith.mulf %[[IN_11]], %[[IN_13]] : f32
// CHECK:                  %[[D27]] = arith.mulf %[[IN]], %[[D26]] : f32
// CHECK:                  %[[D28]] = arith.divf %[[D27]], %[[IN_12]] : f32
// CHECK:                  linalg.yield %[[D28]] : f32
// CHECK:                } -> tensor<?x64xf32>
// CHECK:                %[[D25:.+]] = linalg.matmul ins(%[[D22]], %[[EXTRACTED_SLICE_5]] : tensor<?x?xf32>,
// CHECK-SAME:             tensor<?x64xf32>) outs(%[[D24]] : tensor<?x64xf32>) -> tensor<?x64xf32>
// CHECK:                %[[INSERTED_SLICE_8:.+]] = tensor.insert_slice %[[D25]] into %[[ARG10]][%[[ARG7]], 0, 0] [1,
// CHECK-SAME:             %[[D4]], 64] [1, 1, 1] : tensor<?x64xf32> into tensor<?x?x64xf32>
// CHECK:                %[[INSERTED_SLICE_9:.+]] = tensor.insert_slice %[[D18]] into %[[D6]][0] [%[[D4]]] [1] :
// CHECK-SAME:             tensor<?xf32> into tensor<?xf32>
// CHECK:                %[[INSERTED_SLICE_10:.+]] = tensor.insert_slice %[[D21]] into %[[D8]][0] [%[[D4]]] [1] :
// CHECK-SAME:             tensor<?xf32> into tensor<?xf32>
// CHECK:                scf.yield %[[INSERTED_SLICE_8]], %[[INSERTED_SLICE_9]], %[[INSERTED_SLICE_10]] :
// CHECK-SAME:             tensor<?x?x64xf32>, tensor<?xf32>, tensor<?xf32>
// CHECK:              }
// CHECK:              scf.yield %[[D10]]#[[D0:.+]] : tensor<?x?x64xf32>
// CHECK:            }
// CHECK:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D5]] into %[[ARG6]][%[[D2]], %[[D4]], 0]
// CHECK-SAME:         [%[[ARG3]], %[[ARG5]], 64] [1, 1, 1] : tensor<?x?x64xf32> into tensor<192x1024x64xf32>
// CHECK:            scf.yield %[[INSERTED_SLICE]] : tensor<192x1024x64xf32>
// CHECK:          }
// CHECK:          scf.yield %[[D3]] : tensor<192x1024x64xf32>
// CHECK:        }
// CHECK:        return %[[D1]] : tensor<192x1024x64xf32>
// CHECK:      }
