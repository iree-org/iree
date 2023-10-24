// RUN: iree-dialects-opt --pass-pipeline='builtin.module(func.func(iree-linalg-ext-tile-and-decompose{onlyTile=true}, cse))' --split-input-file %s | FileCheck %s --check-prefix=TILING
// RUN: iree-dialects-opt --pass-pipeline='builtin.module(func.func(iree-linalg-ext-tile-and-decompose{onlyTile=false}, cse))' --split-input-file %s | FileCheck %s --check-prefix=DECOMP
// RUN: iree-dialects-opt --pass-pipeline='builtin.module(func.func(iree-linalg-ext-tile-and-decompose{pipeline=CPU}, cse))' --split-input-file %s | FileCheck %s --check-prefix=CPU
// RUN: iree-dialects-opt --pass-pipeline='builtin.module(func.func(iree-linalg-ext-tile-and-decompose{pipeline=GPU}, cse))' --split-input-file %s | FileCheck %s --check-prefix=GPU
// RUN: iree-dialects-opt --pass-pipeline='builtin.module(func.func(iree-linalg-ext-tile-and-decompose{pipeline=SPIRV}, cse))' --split-input-file %s | FileCheck %s --check-prefix=SPIRV

func.func @attention(%query: tensor<1x1024x64xf32>, %key: tensor<1x1024x64xf32>, %value: tensor<1x1024x64xf32>) -> tensor<1x1024x64xf32> {
  %0 = tensor.empty() : tensor<1x1024x64xf32>
  %1 = iree_linalg_ext.attention ins(%query, %key, %value : tensor<1x1024x64xf32>, tensor<1x1024x64xf32>, tensor<1x1024x64xf32>) outs(%0 : tensor<1x1024x64xf32>) -> tensor<1x1024x64xf32>
  return %1 : tensor<1x1024x64xf32>
}
// TILING-LABEL: @attention
// TILING-SAME: (%[[QUERY:.+]]: tensor<1x1024x64xf32>, %[[KEY:.+]]: tensor<1x1024x64xf32>, %[[VALUE:.+]]: tensor<1x1024x64xf32>)
// TILING:        %[[D0:.+]] = tensor.empty() : tensor<1x1024x64xf32>
// TILING:        %[[D1:.+]] = tensor.empty() : tensor<1024x64xf32>
// TILING-DAG:    %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// TILING:        %[[D2:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D1]] : tensor<1024x64xf32>) ->
// TILING-SAME:     tensor<1024x64xf32>
// TILING-DAG:    %[[CST_0:.+]] = arith.constant -1.000000e+30 : f32
// TILING:        %[[D3:.+]] = tensor.empty() : tensor<1024xf32>
// TILING:        %[[D4:.+]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[D3]] : tensor<1024xf32>) -> tensor<1024xf32>
// TILING:        %[[D5:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D3]] : tensor<1024xf32>) -> tensor<1024xf32>
// TILING-DAG:    %[[C0:.+]] = arith.constant 0 : index
// TILING-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
// TILING:        %[[D6:.+]]:3 = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1024]] step %[[C1024]]
// TILING-SAME:     iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[D2]], %[[ARG5:[a-zA-Z0-9_]+]] = %[[D4]],
// TILING-SAME:     %[[ARG6:[a-zA-Z0-9_]+]] = %[[D5]]) -> (tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>) {
// TILING:          %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[KEY]][0, %[[ARG3]], 0] [1, 1024, 64] [1, 1, 1] :
// TILING-SAME:       tensor<1x1024x64xf32> to tensor<1024x64xf32>
// TILING:          %[[EXTRACTED_SLICE_1:.+]] = tensor.extract_slice %[[VALUE]][0, %[[ARG3]], 0] [1, 1024, 64] [1, 1, 1] :
// TILING-SAME:       tensor<1x1024x64xf32> to tensor<1024x64xf32>
// TILING:          %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[QUERY]][0, 0, 0] [1, 1024, 64] [1, 1, 1] :
// TILING-SAME:       tensor<1x1024x64xf32> to tensor<1024x64xf32>
// TILING:          %[[TILED_ATTENTION:.+]]:3 = iree_linalg_ext.attention ins(%[[EXTRACTED_SLICE_2]], %[[EXTRACTED_SLICE]], %[[EXTRACTED_SLICE_1]] :
// TILING-SAME:                                           outs(%[[ARG4]], %[[ARG5]], %[[ARG6]] :
// TILING-SAME:                                           -> tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>
// TILING:          scf.yield %[[TILED_ATTENTION]]#0, %[[TILED_ATTENTION]]#1, %[[TILED_ATTENTION]]#2 : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>
// TILING:        }
// TILING:        %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D6]]#[[D0:.+]] into %[[D0]][0, 0, 0] [1, 1024, 64] [1,
// TILING-SAME:     1, 1] : tensor<1024x64xf32> into tensor<1x1024x64xf32>
// TILING:        return %[[INSERTED_SLICE]] : tensor<1x1024x64xf32>
// TILING:      }

// DECOMP-DAG:  #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// DECOMP-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
// DECOMP-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0)>
// DECOMP:      func.func @attention(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x1024x64xf32>, %[[ARG1:[a-zA-Z0-9_]+]]:
// DECOMP-SAME:   tensor<1x1024x64xf32>, %[[ARG2:[a-zA-Z0-9_]+]]: tensor<1x1024x64xf32>) -> tensor<1x1024x64xf32> {
// DECOMP:        %[[D0:.+]] = tensor.empty() : tensor<1x1024x64xf32>
// DECOMP:        %[[D1:.+]] = tensor.empty() : tensor<1024x64xf32>
// DECOMP-DAG:    %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// DECOMP:        %[[D2:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D1]] : tensor<1024x64xf32>) ->
// DECOMP-SAME:     tensor<1024x64xf32>
// DECOMP-DAG:    %[[CST_0:.+]] = arith.constant -1.000000e+30 : f32
// DECOMP:        %[[D3:.+]] = tensor.empty() : tensor<1024xf32>
// DECOMP:        %[[D4:.+]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[D3]] : tensor<1024xf32>) -> tensor<1024xf32>
// DECOMP:        %[[D5:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D3]] : tensor<1024xf32>) -> tensor<1024xf32>
// DECOMP-DAG:    %[[C0:.+]] = arith.constant 0 : index
// DECOMP-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
// DECOMP:        %[[D6:.+]]:3 = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1024]] step %[[C1024]]
// DECOMP-SAME:     iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[D2]], %[[ARG5:[a-zA-Z0-9_]+]] = %[[D4]],
// DECOMP-SAME:     %[[ARG6:[a-zA-Z0-9_]+]] = %[[D5]]) -> (tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>) {
// DECOMP:          %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG1]][0, %[[ARG3]], 0] [1, 1024, 64] [1, 1, 1] :
// DECOMP-SAME:       tensor<1x1024x64xf32> to tensor<1024x64xf32>
// DECOMP:          %[[EXTRACTED_SLICE_1:.+]] = tensor.extract_slice %[[ARG2]][0, %[[ARG3]], 0] [1, 1024, 64] [1, 1, 1] :
// DECOMP-SAME:       tensor<1x1024x64xf32> to tensor<1024x64xf32>
// DECOMP:          %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[ARG0]][0, 0, 0] [1, 1024, 64] [1, 1, 1] :
// DECOMP-SAME:       tensor<1x1024x64xf32> to tensor<1024x64xf32>
// DECOMP:          %[[D7:.+]] = tensor.empty() : tensor<1024x1024xf32>
// DECOMP:          %[[D8:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D7]] : tensor<1024x1024xf32>) ->
// DECOMP-SAME:       tensor<1024x1024xf32>
// DECOMP:          %[[D9:.+]] = linalg.matmul_transpose_b ins(%[[EXTRACTED_SLICE_2]], %[[EXTRACTED_SLICE]] :
// DECOMP-SAME:       tensor<1024x64xf32>, tensor<1024x64xf32>) outs(%[[D8]] : tensor<1024x1024xf32>) ->
// DECOMP-SAME:       tensor<1024x1024xf32>
// DECOMP:          %[[D10:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]]], iterator_types = ["parallel",
// DECOMP-SAME:       "reduction"]} ins(%[[D9]] : tensor<1024x1024xf32>) outs(%[[ARG5]] : tensor<1024xf32>) {
// DECOMP:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// DECOMP:            %[[D18:.+]] = arith.maximumf %[[IN]], %[[OUT]] : f32
// DECOMP:            linalg.yield %[[D18]] : f32
// DECOMP:          } -> tensor<1024xf32>
// DECOMP:          %[[D11:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP]]], iterator_types = ["parallel",
// DECOMP-SAME:       "parallel"]} ins(%[[D10]] : tensor<1024xf32>) outs(%[[D9]] : tensor<1024x1024xf32>) {
// DECOMP:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// DECOMP:            %[[D18]] = arith.subf %[[OUT]], %[[IN]] : f32
// DECOMP:            %[[D19:.+]] = math.exp2 %[[D18]] : f32
// DECOMP:            linalg.yield %[[D19]] : f32
// DECOMP:          } -> tensor<1024x1024xf32>
// DECOMP:          %[[D12:.+]] = linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP2]], #[[MAP2]]], iterator_types =
// DECOMP-SAME:       ["parallel"]} ins(%[[ARG5]], %[[D10]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[ARG6]] :
// DECOMP-SAME:       tensor<1024xf32>) {
// DECOMP:          ^bb0(%[[IN:.+]]: f32, %[[IN_3:.+]]: f32, %[[OUT:.+]]: f32):
// DECOMP:            %[[D18]] = arith.subf %[[IN]], %[[IN_3]] : f32
// DECOMP:            %[[D19]] = math.exp2 %[[D18]] : f32
// DECOMP:            %[[D20:.+]] = arith.mulf %[[D19]], %[[OUT]] : f32
// DECOMP:            linalg.yield %[[D20]] : f32
// DECOMP:          } -> tensor<1024xf32>
// DECOMP:          %[[D13:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]]], iterator_types = ["parallel",
// DECOMP-SAME:       "reduction"]} ins(%[[D11]] : tensor<1024x1024xf32>) outs(%[[D12]] : tensor<1024xf32>) {
// DECOMP:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// DECOMP:            %[[D18]] = arith.addf %[[IN]], %[[OUT]] : f32
// DECOMP:            linalg.yield %[[D18]] : f32
// DECOMP:          } -> tensor<1024xf32>
// DECOMP:          %[[D14:.+]] = linalg.generic {indexing_maps = [#[[MAP2]]], iterator_types = ["parallel"]}
// DECOMP-SAME:       outs(%[[D13]] : tensor<1024xf32>) {
// DECOMP:          ^bb0(%[[OUT:.+]]: f32):
// DECOMP-DAG:        %[[CST_3:.+]] = arith.constant 1.000000e+00 : f32
// DECOMP:            %[[D18]] = arith.divf %[[CST_3]], %[[OUT]] : f32
// DECOMP:            linalg.yield %[[D18]] : f32
// DECOMP:          } -> tensor<1024xf32>
// DECOMP:          %[[D15:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP]]], iterator_types = ["parallel",
// DECOMP-SAME:       "parallel"]} ins(%[[D14]] : tensor<1024xf32>) outs(%[[D11]] : tensor<1024x1024xf32>) {
// DECOMP:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// DECOMP:            %[[D18]] = arith.mulf %[[OUT]], %[[IN]] : f32
// DECOMP:            linalg.yield %[[D18]] : f32
// DECOMP:          } -> tensor<1024x1024xf32>
// DECOMP:          %[[D16:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP1]], #[[MAP]]], iterator_types =
// DECOMP-SAME:       ["parallel", "parallel"]} ins(%[[D12]], %[[D14]] : tensor<1024xf32>, tensor<1024xf32>)
// DECOMP-SAME:       outs(%[[ARG4]] : tensor<1024x64xf32>) {
// DECOMP:          ^bb0(%[[IN:.+]]: f32, %[[IN_3:.+]]: f32, %[[OUT:.+]]: f32):
// DECOMP:            %[[D18]] = arith.mulf %[[IN]], %[[IN_3]] : f32
// DECOMP:            %[[D19]] = arith.mulf %[[D18]], %[[OUT]] : f32
// DECOMP:            linalg.yield %[[D19]] : f32
// DECOMP:          } -> tensor<1024x64xf32>
// DECOMP:          %[[D17:.+]] = linalg.matmul ins(%[[D15]], %[[EXTRACTED_SLICE_1]] : tensor<1024x1024xf32>,
// DECOMP-SAME:       tensor<1024x64xf32>) outs(%[[D16]] : tensor<1024x64xf32>) -> tensor<1024x64xf32>
// DECOMP:          scf.yield %[[D17]], %[[D10]], %[[D13]] : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>
// DECOMP:        }
// DECOMP:        %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D6]]#[[D0:.+]] into %[[D0]][0, 0, 0] [1, 1024, 64] [1,
// DECOMP-SAME:     1, 1] : tensor<1024x64xf32> into tensor<1x1024x64xf32>
// DECOMP:        return %[[INSERTED_SLICE]] : tensor<1x1024x64xf32>
// DECOMP:      }

// CPU:      func.func @attention
// CPU-NOT:       iree_linalg_ext.attention

// GPU:      func.func @attention
// GPU-NOT:       iree_linalg_ext.attention

// SPIRV:    func.func @attention
// SPIRV:         iree_linalg_ext.attention

// -----

func.func @attention(%query: tensor<?x?x?xf32>, %key: tensor<?x?x?xf32>, %value: tensor<?x?x?xf32>, %dim0: index, %dim1: index, %dim2: index) -> tensor<?x?x?xf32> {
  %0 = tensor.empty(%dim0, %dim1, %dim2) : tensor<?x?x?xf32>
  %1 = iree_linalg_ext.attention ins(%query, %key, %value : tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}
// TILING:      @attention(
// TILING-SAME:  %[[QUERY:.+]]: tensor<?x?x?xf32>, %[[KEY:.+]]: tensor<?x?x?xf32>, %[[VALUE:.+]]: tensor<?x?x?xf32>,
// TILING-SAME:  %[[ARG3:[a-zA-Z0-9_]+]]: index, %[[ARG4:[a-zA-Z0-9_]+]]: index, %[[ARG5:[a-zA-Z0-9_]+]]: index)
// TILING:        %[[D0:.+]] = tensor.empty(%[[ARG3]], %[[ARG4]], %[[ARG5]]) : tensor<?x?x?xf32>
// TILING-DAG:    %[[C0:.+]] = arith.constant 0 : index
// TILING-DAG:    %[[C1:.+]] = arith.constant 1 : index
// TILING:        %[[DIM:.+]] = tensor.dim %[[QUERY]], %[[C1]] : tensor<?x?x?xf32>
// TILING-DAG:    %[[C2:.+]] = arith.constant 2 : index
// TILING:        %[[DIM_0:.+]] = tensor.dim %[[QUERY]], %[[C2]] : tensor<?x?x?xf32>
// TILING:        %[[DIM_1:.+]] = tensor.dim %[[KEY]], %[[C1]] : tensor<?x?x?xf32>
// TILING:        %[[D1:.+]] = tensor.empty(%[[DIM]], %[[DIM_0]]) : tensor<?x?xf32>
// TILING-DAG:    %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// TILING:        %[[D2:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D1]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// TILING-DAG:    %[[CST_2:.+]] = arith.constant -1.000000e+30 : f32
// TILING:        %[[D3:.+]] = tensor.empty(%[[DIM]]) : tensor<?xf32>
// TILING:        %[[D4:.+]] = linalg.fill ins(%[[CST_2]] : f32) outs(%[[D3]] : tensor<?xf32>) -> tensor<?xf32>
// TILING:        %[[D5:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D3]] : tensor<?xf32>) -> tensor<?xf32>
// TILING:        %[[D6:.+]]:3 = scf.for %[[ARG6:[a-zA-Z0-9_]+]] = %[[C0]] to %[[DIM_1]] step %[[DIM]]
// TILING-SAME:     iter_args(%[[ARG7:[a-zA-Z0-9_]+]] = %[[D2]], %[[ARG8:[a-zA-Z0-9_]+]] = %[[D4]],
// TILING-SAME:     %[[ARG9:[a-zA-Z0-9_]+]] = %[[D5]]) -> (tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>) {
// TILING:          %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[KEY]][0, %[[ARG6]], 0] [1, %[[DIM]], %[[DIM_0]]]
// TILING-SAME:       [1, 1, 1] : tensor<?x?x?xf32> to tensor<?x?xf32>
// TILING:          %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[VALUE]][0, %[[ARG6]], 0] [1, %[[DIM]], %[[DIM_0]]]
// TILING-SAME:       [1, 1, 1] : tensor<?x?x?xf32> to tensor<?x?xf32>
// TILING:          %[[EXTRACTED_SLICE_4:.+]] = tensor.extract_slice %[[QUERY]][0, 0, 0] [1, %[[DIM]], %[[DIM_0]]] [1, 1,
// TILING-SAME:       1] : tensor<?x?x?xf32> to tensor<?x?xf32>
// TILING:          %[[TILED_ATTENTION]]:3 = iree_linalg_ext.attention ins(%[[EXTRACTED_SLICE_4]], %[[EXTRACTED_SLICE]], %[[EXTRACTED_SLICE_3]] : 
// TILING-SAME:                      outs(%[[ARG7]], %[[ARG8]], %[[ARG9]] : 
// TILING-SAME:                      -> tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>
// TILING:          scf.yield %[[TILED_ATTENTION]]#0, %[[TILED_ATTENTION]]#1, %[[TILED_ATTENTION]]#2 : tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>
// TILING:        }
// TILING:        %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D6]]#[[D0:.+]] into %[[D0]][0, 0, 0] [1, %[[DIM]],
// TILING-SAME:     %[[DIM_0]]] [1, 1, 1] : tensor<?x?xf32> into tensor<?x?x?xf32>
// TILING:        return %[[INSERTED_SLICE]] : tensor<?x?x?xf32>
// TILING:      }

// DECOMP-DAG:  #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// DECOMP-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
// DECOMP-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0)>
// DECOMP:      func.func @attention(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>, %[[ARG1:[a-zA-Z0-9_]+]]:
// DECOMP-SAME:   tensor<?x?x?xf32>, %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>, %[[ARG3:[a-zA-Z0-9_]+]]: index,
// DECOMP-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: index, %[[ARG5:[a-zA-Z0-9_]+]]: index) -> tensor<?x?x?xf32> {
// DECOMP:        %[[D0:.+]] = tensor.empty(%[[ARG3]], %[[ARG4]], %[[ARG5]]) : tensor<?x?x?xf32>
// DECOMP-DAG:    %[[C0:.+]] = arith.constant 0 : index
// DECOMP-DAG:    %[[C1:.+]] = arith.constant 1 : index
// DECOMP:        %[[DIM:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?xf32>
// DECOMP-DAG:    %[[C2:.+]] = arith.constant 2 : index
// DECOMP:        %[[DIM_0:.+]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?xf32>
// DECOMP:        %[[DIM_1:.+]] = tensor.dim %[[ARG1]], %[[C1]] : tensor<?x?x?xf32>
// DECOMP:        %[[D1:.+]] = tensor.empty(%[[DIM]], %[[DIM_0]]) : tensor<?x?xf32>
// DECOMP-DAG:    %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// DECOMP:        %[[D2:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D1]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// DECOMP-DAG:    %[[CST_2:.+]] = arith.constant -1.000000e+30 : f32
// DECOMP:        %[[D3:.+]] = tensor.empty(%[[DIM]]) : tensor<?xf32>
// DECOMP:        %[[D4:.+]] = linalg.fill ins(%[[CST_2]] : f32) outs(%[[D3]] : tensor<?xf32>) -> tensor<?xf32>
// DECOMP:        %[[D5:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D3]] : tensor<?xf32>) -> tensor<?xf32>
// DECOMP:        %[[D6:.+]]:3 = scf.for %[[ARG6:[a-zA-Z0-9_]+]] = %[[C0]] to %[[DIM_1]] step %[[DIM]]
// DECOMP-SAME:     iter_args(%[[ARG7:[a-zA-Z0-9_]+]] = %[[D2]], %[[ARG8:[a-zA-Z0-9_]+]] = %[[D4]],
// DECOMP-SAME:     %[[ARG9:[a-zA-Z0-9_]+]] = %[[D5]]) -> (tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>) {
// DECOMP:          %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG1]][0, %[[ARG6]], 0] [1, %[[DIM]], %[[DIM_0]]]
// DECOMP-SAME:       [1, 1, 1] : tensor<?x?x?xf32> to tensor<?x?xf32>
// DECOMP:          %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[ARG2]][0, %[[ARG6]], 0] [1, %[[DIM]], %[[DIM_0]]]
// DECOMP-SAME:       [1, 1, 1] : tensor<?x?x?xf32> to tensor<?x?xf32>
// DECOMP:          %[[EXTRACTED_SLICE_4:.+]] = tensor.extract_slice %[[ARG0]][0, 0, 0] [1, %[[DIM]], %[[DIM_0]]] [1, 1,
// DECOMP-SAME:       1] : tensor<?x?x?xf32> to tensor<?x?xf32>
// DECOMP:          %[[QUERY_SLICE_DIM_0:.+]] = tensor.dim %[[EXTRACTED_SLICE_4]], %[[C0]] : tensor<?x?xf32>
// DECOMP:          %[[D7:.+]] = tensor.empty(%[[QUERY_SLICE_DIM_0]], %[[QUERY_SLICE_DIM_0]]) : tensor<?x?xf32>
// DECOMP:          %[[D8:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D7]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// DECOMP:          %[[D9:.+]] = linalg.matmul_transpose_b ins(%[[EXTRACTED_SLICE_4]], %[[EXTRACTED_SLICE]] :
// DECOMP-SAME:       tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[D8]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// DECOMP:          %[[D10:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]]], iterator_types = ["parallel",
// DECOMP-SAME:       "reduction"]} ins(%[[D9]] : tensor<?x?xf32>) outs(%[[ARG8]] : tensor<?xf32>) {
// DECOMP:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// DECOMP:            %[[D18:.+]] = arith.maximumf %[[IN]], %[[OUT]] : f32
// DECOMP:            linalg.yield %[[D18]] : f32
// DECOMP:          } -> tensor<?xf32>
// DECOMP:          %[[D11:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP]]], iterator_types = ["parallel",
// DECOMP-SAME:       "parallel"]} ins(%[[D10]] : tensor<?xf32>) outs(%[[D9]] : tensor<?x?xf32>) {
// DECOMP:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// DECOMP:            %[[D18]] = arith.subf %[[OUT]], %[[IN]] : f32
// DECOMP:            %[[D19:.+]] = math.exp2 %[[D18]] : f32
// DECOMP:            linalg.yield %[[D19]] : f32
// DECOMP:          } -> tensor<?x?xf32>
// DECOMP:          %[[D12:.+]] = linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP2]], #[[MAP2]]], iterator_types =
// DECOMP-SAME:       ["parallel"]} ins(%[[ARG8]], %[[D10]] : tensor<?xf32>, tensor<?xf32>) outs(%[[ARG9]] :
// DECOMP-SAME:       tensor<?xf32>) {
// DECOMP:          ^bb0(%[[IN:.+]]: f32, %[[IN_5:.+]]: f32, %[[OUT:.+]]: f32):
// DECOMP:            %[[D18]] = arith.subf %[[IN]], %[[IN_5]] : f32
// DECOMP:            %[[D19]] = math.exp2 %[[D18]] : f32
// DECOMP:            %[[D20:.+]] = arith.mulf %[[D19]], %[[OUT]] : f32
// DECOMP:            linalg.yield %[[D20]] : f32
// DECOMP:          } -> tensor<?xf32>
// DECOMP:          %[[D13:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]]], iterator_types = ["parallel",
// DECOMP-SAME:       "reduction"]} ins(%[[D11]] : tensor<?x?xf32>) outs(%[[D12]] : tensor<?xf32>) {
// DECOMP:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// DECOMP:            %[[D18]] = arith.addf %[[IN]], %[[OUT]] : f32
// DECOMP:            linalg.yield %[[D18]] : f32
// DECOMP:          } -> tensor<?xf32>
// DECOMP:          %[[D14:.+]] = linalg.generic {indexing_maps = [#[[MAP2]]], iterator_types = ["parallel"]}
// DECOMP-SAME:       outs(%[[D13]] : tensor<?xf32>) {
// DECOMP:          ^bb0(%[[OUT:.+]]: f32):
// DECOMP-DAG:        %[[CST_5:.+]] = arith.constant 1.000000e+00 : f32
// DECOMP:            %[[D18]] = arith.divf %[[CST_5]], %[[OUT]] : f32
// DECOMP:            linalg.yield %[[D18]] : f32
// DECOMP:          } -> tensor<?xf32>
// DECOMP:          %[[D15:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP]]], iterator_types = ["parallel",
// DECOMP-SAME:       "parallel"]} ins(%[[D14]] : tensor<?xf32>) outs(%[[D11]] : tensor<?x?xf32>) {
// DECOMP:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// DECOMP:            %[[D18]] = arith.mulf %[[OUT]], %[[IN]] : f32
// DECOMP:            linalg.yield %[[D18]] : f32
// DECOMP:          } -> tensor<?x?xf32>
// DECOMP:          %[[D16:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP1]], #[[MAP]]], iterator_types =
// DECOMP-SAME:       ["parallel", "parallel"]} ins(%[[D12]], %[[D14]] : tensor<?xf32>, tensor<?xf32>) outs(%[[ARG7]] :
// DECOMP-SAME:       tensor<?x?xf32>) {
// DECOMP:          ^bb0(%[[IN:.+]]: f32, %[[IN_5:.+]]: f32, %[[OUT:.+]]: f32):
// DECOMP:            %[[D18]] = arith.mulf %[[IN]], %[[IN_5]] : f32
// DECOMP:            %[[D19]] = arith.mulf %[[D18]], %[[OUT]] : f32
// DECOMP:            linalg.yield %[[D19]] : f32
// DECOMP:          } -> tensor<?x?xf32>
// DECOMP:          %[[D17:.+]] = linalg.matmul ins(%[[D15]], %[[EXTRACTED_SLICE_3]] : tensor<?x?xf32>, tensor<?x?xf32>)
// DECOMP-SAME:       outs(%[[D16]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// DECOMP:          scf.yield %[[D17]], %[[D10]], %[[D13]] : tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>
// DECOMP:        }
// DECOMP:        %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D6]]#[[D0:.+]] into %[[D0]][0, 0, 0] [1, %[[DIM]],
// DECOMP-SAME:     %[[DIM_0]]] [1, 1, 1] : tensor<?x?xf32> into tensor<?x?x?xf32>
// DECOMP:        return %[[INSERTED_SLICE]] : tensor<?x?x?xf32>
// DECOMP:      }

// CPU:      func.func @attention
// CPU-NOT:       iree_linalg_ext.attention

// GPU:      func.func @attention
// GPU-NOT:       iree_linalg_ext.attention

// SPIRV:    func.func @attention
// SPIRV:         iree_linalg_ext.attention

// -----

func.func @attention(%query: tensor<1x1024x64xf16>, %key: tensor<1x1024x64xf16>, %value: tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16> {
  %0 = tensor.empty() : tensor<1x1024x64xf16>
  %1 = iree_linalg_ext.attention ins(%query, %key, %value : tensor<1x1024x64xf16>, tensor<1x1024x64xf16>, tensor<1x1024x64xf16>) outs(%0 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
  return %1 : tensor<1x1024x64xf16>
}
// TILING-DAG:  #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// TILING:      @attention(
// TILING-SAME:  %[[QUERY:.+]]: tensor<1x1024x64xf16>, %[[KEY:.+]]: tensor<1x1024x64xf16>, %[[VALUE:.+]]: tensor<1x1024x64xf16>)
// TILING:        %[[D0:.+]] = tensor.empty() : tensor<1x1024x64xf16>
// TILING:        %[[D1:.+]] = tensor.empty() : tensor<1024x64xf32>
// TILING:        %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[D0]][0, 0, 0] [1, 1024, 64] [1, 1, 1] :
// TILING-SAME:     tensor<1x1024x64xf16> to tensor<1024x64xf16>
// TILING-DAG:    %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// TILING:        %[[D2:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D1]] : tensor<1024x64xf32>) ->
// TILING-SAME:     tensor<1024x64xf32>
// TILING-DAG:    %[[CST_0:.+]] = arith.constant -1.000000e+30 : f32
// TILING:        %[[D3:.+]] = tensor.empty() : tensor<1024xf32>
// TILING:        %[[D4:.+]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[D3]] : tensor<1024xf32>) -> tensor<1024xf32>
// TILING:        %[[D5:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D3]] : tensor<1024xf32>) -> tensor<1024xf32>
// TILING-DAG:    %[[C0:.+]] = arith.constant 0 : index
// TILING-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
// TILING:        %[[D6:.+]]:3 = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1024]] step %[[C1024]]
// TILING-SAME:     iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[D2]], %[[ARG5:[a-zA-Z0-9_]+]] = %[[D4]],
// TILING-SAME:     %[[ARG6:[a-zA-Z0-9_]+]] = %[[D5]]) -> (tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>) {
// TILING:          %[[EXTRACTED_SLICE_1:.+]] = tensor.extract_slice %[[KEY]][0, %[[ARG3]], 0] [1, 1024, 64] [1, 1, 1] :
// TILING-SAME:       tensor<1x1024x64xf16> to tensor<1024x64xf16>
// TILING:          %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[VALUE]][0, %[[ARG3]], 0] [1, 1024, 64] [1, 1, 1] :
// TILING-SAME:       tensor<1x1024x64xf16> to tensor<1024x64xf16>
// TILING:          %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[QUERY]][0, 0, 0] [1, 1024, 64] [1, 1, 1] :
// TILING-SAME:       tensor<1x1024x64xf16> to tensor<1024x64xf16>
// TILING:          %[[TILED_ATTENTION:.+]]:3 = iree_linalg_ext.attention ins(%[[EXTRACTED_SLICE_3]], %[[EXTRACTED_SLICE_1]], %[[EXTRACTED_SLICE_2]] :
// TILING-SAME:                                           outs(%[[ARG4]], %[[ARG5]], %[[ARG6]] :
// TILING-SAME:                                           -> tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>
// TILING:          scf.yield %[[TILED_ATTENTION]]#0, %[[TILED_ATTENTION]]#1, %[[TILED_ATTENTION]]#2 : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>
// TILING:        }
// TILING:        %[[D7:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel",
// TILING-SAME:     "parallel"]} ins(%[[D6]]#[[D0:.+]] : tensor<1024x64xf32>) outs(%[[EXTRACTED_SLICE]] :
// TILING-SAME:     tensor<1024x64xf16>) {
// TILING:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f16):
// TILING:          %[[D8:.+]] = arith.truncf %[[IN]] : f32 to f16
// TILING:          linalg.yield %[[D8]] : f16
// TILING:        } -> tensor<1024x64xf16>
// TILING:        %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D7]] into %[[D0]][0, 0, 0] [1, 1024, 64] [1, 1, 1] :
// TILING-SAME:     tensor<1024x64xf16> into tensor<1x1024x64xf16>
// TILING:        return %[[INSERTED_SLICE]] : tensor<1x1024x64xf16>
// TILING:      }

// DECOMP-DAG:  #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// DECOMP-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
// DECOMP-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0)>
// DECOMP:      func.func @attention(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x1024x64xf16>, %[[ARG1:[a-zA-Z0-9_]+]]:
// DECOMP-SAME:   tensor<1x1024x64xf16>, %[[ARG2:[a-zA-Z0-9_]+]]: tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16> {
// DECOMP:        %[[D0:.+]] = tensor.empty() : tensor<1x1024x64xf16>
// DECOMP:        %[[D1:.+]] = tensor.empty() : tensor<1024x64xf32>
// DECOMP:        %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[D0]][0, 0, 0] [1, 1024, 64] [1, 1, 1] :
// DECOMP-SAME:     tensor<1x1024x64xf16> to tensor<1024x64xf16>
// DECOMP-DAG:    %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// DECOMP:        %[[D2:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D1]] : tensor<1024x64xf32>) ->
// DECOMP-SAME:     tensor<1024x64xf32>
// DECOMP-DAG:    %[[CST_0:.+]] = arith.constant -1.000000e+30 : f32
// DECOMP:        %[[D3:.+]] = tensor.empty() : tensor<1024xf32>
// DECOMP:        %[[D4:.+]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[D3]] : tensor<1024xf32>) -> tensor<1024xf32>
// DECOMP:        %[[D5:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D3]] : tensor<1024xf32>) -> tensor<1024xf32>
// DECOMP-DAG:    %[[C0:.+]] = arith.constant 0 : index
// DECOMP-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
// DECOMP:        %[[D6:.+]]:3 = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1024]] step %[[C1024]]
// DECOMP-SAME:     iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[D2]], %[[ARG5:[a-zA-Z0-9_]+]] = %[[D4]],
// DECOMP-SAME:     %[[ARG6:[a-zA-Z0-9_]+]] = %[[D5]]) -> (tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>) {
// DECOMP:          %[[EXTRACTED_SLICE_1:.+]] = tensor.extract_slice %[[ARG1]][0, %[[ARG3]], 0] [1, 1024, 64] [1, 1, 1] :
// DECOMP-SAME:       tensor<1x1024x64xf16> to tensor<1024x64xf16>
// DECOMP:          %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[ARG2]][0, %[[ARG3]], 0] [1, 1024, 64] [1, 1, 1] :
// DECOMP-SAME:       tensor<1x1024x64xf16> to tensor<1024x64xf16>
// DECOMP:          %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[ARG0]][0, 0, 0] [1, 1024, 64] [1, 1, 1] :
// DECOMP-SAME:       tensor<1x1024x64xf16> to tensor<1024x64xf16>
// DECOMP:          %[[D8:.+]] = tensor.empty() : tensor<1024x1024xf32>
// DECOMP:          %[[D9:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D8]] : tensor<1024x1024xf32>) ->
// DECOMP-SAME:       tensor<1024x1024xf32>
// DECOMP:          %[[D10:.+]] = linalg.matmul_transpose_b ins(%[[EXTRACTED_SLICE_3]], %[[EXTRACTED_SLICE_1]] :
// DECOMP-SAME:       tensor<1024x64xf16>, tensor<1024x64xf16>) outs(%[[D9]] : tensor<1024x1024xf32>) ->
// DECOMP-SAME:       tensor<1024x1024xf32>
// DECOMP:          %[[D11:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]]], iterator_types = ["parallel",
// DECOMP-SAME:       "reduction"]} ins(%[[D10]] : tensor<1024x1024xf32>) outs(%[[ARG5]] : tensor<1024xf32>) {
// DECOMP:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// DECOMP:            %[[D21:.+]] = arith.maximumf %[[IN]], %[[OUT]] : f32
// DECOMP:            linalg.yield %[[D21]] : f32
// DECOMP:          } -> tensor<1024xf32>
// DECOMP:          %[[D12:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP]]], iterator_types = ["parallel",
// DECOMP-SAME:       "parallel"]} ins(%[[D11]] : tensor<1024xf32>) outs(%[[D10]] : tensor<1024x1024xf32>) {
// DECOMP:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// DECOMP:            %[[D21]] = arith.subf %[[OUT]], %[[IN]] : f32
// DECOMP:            %[[D22:.+]] = math.exp2 %[[D21]] : f32
// DECOMP:            linalg.yield %[[D22]] : f32
// DECOMP:          } -> tensor<1024x1024xf32>
// DECOMP:          %[[D13:.+]] = linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP2]], #[[MAP2]]], iterator_types =
// DECOMP-SAME:       ["parallel"]} ins(%[[ARG5]], %[[D11]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[ARG6]] :
// DECOMP-SAME:       tensor<1024xf32>) {
// DECOMP:          ^bb0(%[[IN:.+]]: f32, %[[IN_4:.+]]: f32, %[[OUT:.+]]: f32):
// DECOMP:            %[[D21]] = arith.subf %[[IN]], %[[IN_4]] : f32
// DECOMP:            %[[D22]] = math.exp2 %[[D21]] : f32
// DECOMP:            %[[D23:.+]] = arith.mulf %[[D22]], %[[OUT]] : f32
// DECOMP:            linalg.yield %[[D23]] : f32
// DECOMP:          } -> tensor<1024xf32>
// DECOMP:          %[[D14:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]]], iterator_types = ["parallel",
// DECOMP-SAME:       "reduction"]} ins(%[[D12]] : tensor<1024x1024xf32>) outs(%[[D13]] : tensor<1024xf32>) {
// DECOMP:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// DECOMP:            %[[D21]] = arith.addf %[[IN]], %[[OUT]] : f32
// DECOMP:            linalg.yield %[[D21]] : f32
// DECOMP:          } -> tensor<1024xf32>
// DECOMP:          %[[D15:.+]] = linalg.generic {indexing_maps = [#[[MAP2]]], iterator_types = ["parallel"]}
// DECOMP-SAME:       outs(%[[D14]] : tensor<1024xf32>) {
// DECOMP:          ^bb0(%[[OUT:.+]]: f32):
// DECOMP-DAG:        %[[CST_4:.+]] = arith.constant 1.000000e+00 : f32
// DECOMP:            %[[D21]] = arith.divf %[[CST_4]], %[[OUT]] : f32
// DECOMP:            linalg.yield %[[D21]] : f32
// DECOMP:          } -> tensor<1024xf32>
// DECOMP:          %[[D16:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP]]], iterator_types = ["parallel",
// DECOMP-SAME:       "parallel"]} ins(%[[D15]] : tensor<1024xf32>) outs(%[[D12]] : tensor<1024x1024xf32>) {
// DECOMP:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// DECOMP:            %[[D21]] = arith.mulf %[[OUT]], %[[IN]] : f32
// DECOMP:            linalg.yield %[[D21]] : f32
// DECOMP:          } -> tensor<1024x1024xf32>
// DECOMP:          %[[D17:.+]] = tensor.empty() : tensor<1024x1024xf16>
// DECOMP:          %[[D18:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel",
// DECOMP-SAME:       "parallel"]} ins(%[[D16]] : tensor<1024x1024xf32>) outs(%[[D17]] : tensor<1024x1024xf16>) {
// DECOMP:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f16):
// DECOMP:            %[[D21]] = arith.truncf %[[IN]] : f32 to f16
// DECOMP:            linalg.yield %[[D21]] : f16
// DECOMP:          } -> tensor<1024x1024xf16>
// DECOMP:          %[[D19:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP1]], #[[MAP]]], iterator_types =
// DECOMP-SAME:       ["parallel", "parallel"]} ins(%[[D13]], %[[D15]] : tensor<1024xf32>, tensor<1024xf32>)
// DECOMP-SAME:       outs(%[[ARG4]] : tensor<1024x64xf32>) {
// DECOMP:          ^bb0(%[[IN:.+]]: f32, %[[IN_4:.+]]: f32, %[[OUT:.+]]: f32):
// DECOMP:            %[[D21]] = arith.mulf %[[IN]], %[[IN_4]] : f32
// DECOMP:            %[[D22]] = arith.mulf %[[D21]], %[[OUT]] : f32
// DECOMP:            linalg.yield %[[D22]] : f32
// DECOMP:          } -> tensor<1024x64xf32>
// DECOMP:          %[[D20:.+]] = linalg.matmul ins(%[[D18]], %[[EXTRACTED_SLICE_2]] : tensor<1024x1024xf16>,
// DECOMP-SAME:       tensor<1024x64xf16>) outs(%[[D19]] : tensor<1024x64xf32>) -> tensor<1024x64xf32>
// DECOMP:          scf.yield %[[D20]], %[[D11]], %[[D14]] : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>
// DECOMP:        }
// DECOMP:        %[[D7:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel",
// DECOMP-SAME:     "parallel"]} ins(%[[D6]]#[[D0:.+]] : tensor<1024x64xf32>) outs(%[[EXTRACTED_SLICE]] :
// DECOMP-SAME:     tensor<1024x64xf16>) {
// DECOMP:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f16):
// DECOMP:          %[[D8]] = arith.truncf %[[IN]] : f32 to f16
// DECOMP:          linalg.yield %[[D8]] : f16
// DECOMP:        } -> tensor<1024x64xf16>
// DECOMP:        %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D7]] into %[[D0]][0, 0, 0] [1, 1024, 64] [1, 1, 1] :
// DECOMP-SAME:     tensor<1024x64xf16> into tensor<1x1024x64xf16>
// DECOMP:        return %[[INSERTED_SLICE]] : tensor<1x1024x64xf16>
// DECOMP:      }

// CPU:      func.func @attention
// CPU-NOT:       iree_linalg_ext.attention

// GPU:      func.func @attention
// GPU-NOT:       iree_linalg_ext.attention

// SPIRV:    func.func @attention
// SPIRV:         iree_linalg_ext.attention

// -----

#map = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
#map1 = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
module {
  func.func @winograd_input_transform(%arg0: tensor<1x10x10x1280xf32>) -> tensor<8x8x1x2x2x1280xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1280 = arith.constant 1280 : index
    %c32 = arith.constant 32 : index
    %0 = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
    %1 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %0) -> (tensor<8x8x1x2x2x1280xf32>) {
      %2 = affine.min #map(%arg1)[%c1, %c1]
      %3 = scf.for %arg3 = %c0 to %c1280 step %c32 iter_args(%arg4 = %arg2) -> (tensor<8x8x1x2x2x1280xf32>) {
        %4 = affine.min #map1(%arg3)[%c32, %c1280]
        %extracted_slice = tensor.extract_slice %arg0[%arg1, 0, 0, %arg3] [%2, 10, 10, %4] [1, 1, 1, 1] : tensor<1x10x10x1280xf32> to tensor<?x10x10x?xf32>
        %extracted_slice_0 = tensor.extract_slice %0[0, 0, %arg1, 0, 0, %arg3] [8, 8, %2, 2, 2, %4] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x1280xf32> to tensor<8x8x?x2x2x?xf32>
        %5 = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2]) ins(%extracted_slice : tensor<?x10x10x?xf32>) outs(%extracted_slice_0 : tensor<8x8x?x2x2x?xf32>) -> tensor<8x8x?x2x2x?xf32>
        %inserted_slice = tensor.insert_slice %5 into %arg4[0, 0, %arg1, 0, 0, %arg3] [8, 8, %2, 2, 2, %4] [1, 1, 1, 1, 1, 1] : tensor<8x8x?x2x2x?xf32> into tensor<8x8x1x2x2x1280xf32>
        scf.yield %inserted_slice : tensor<8x8x1x2x2x1280xf32>
      }
      scf.yield %3 : tensor<8x8x1x2x2x1280xf32>
    }
    return %1 : tensor<8x8x1x2x2x1280xf32>
  }
}
// TILING-DAG:  #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
// TILING-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// TILING-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 6)>
// TILING-DAG:  #[[MAP3:.+]] = affine_map<(d0) -> (-d0 + 10, 8)>
// TILING:      func.func @winograd_input_transform(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x10x10x1280xf32>) ->
// TILING-SAME:   tensor<8x8x1x2x2x1280xf32> {
// TILING:        %[[C0:.+]] = arith.constant 0 : index
// TILING:        %[[C1:.+]] = arith.constant 1 : index
// TILING:        %[[C2:.+]] = arith.constant 2 : index
// TILING:        %[[C1280:.+]] = arith.constant 1280 : index
// TILING:        %[[C32:.+]] = arith.constant 32 : index
// TILING:        %[[D1:.+]] = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
// TILING:        %[[D2:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1]] step %[[C1]]
// TILING-SAME:     iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D1]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// TILING-DAG:        %[[D3:.+]] = affine.min #[[MAP]](%[[ARG1]])[%[[C1]], %[[C1]]]
// TILING:          %[[D4:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1280]] step %[[C32]]
// TILING-SAME:       iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// TILING-DAG:          %[[D5:.+]] = affine.min #[[MAP1]](%[[ARG3]])[%[[C32]], %[[C1280]]]
// TILING:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG1]], 0, 0, %[[ARG3]]] [%[[D3]], 10,
// TILING-SAME:         10, %[[D5]]] [1, 1, 1, 1] : tensor<1x10x10x1280xf32> to tensor<?x10x10x?xf32>
// TILING:            %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[D1]][0, 0, %[[ARG1]], 0, 0, %[[ARG3]]] [8, 8,
// TILING-SAME:         %[[D3]], 2, 2, %[[D5]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x1280xf32> to
// TILING-SAME:         tensor<8x8x?x2x2x?xf32>
// TILING:            %[[D6:.+]] = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[D3]] step %[[C1]]
// TILING-SAME:         iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[EXTRACTED_SLICE_2]]) -> (tensor<8x8x?x2x2x?xf32>) {
// TILING:              %[[D7:.+]] = scf.for %[[ARG7:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// TILING-SAME:           iter_args(%[[ARG8:[a-zA-Z0-9_]+]] = %[[ARG6]]) -> (tensor<8x8x?x2x2x?xf32>) {
// TILING-DAG:              %[[D8:.+]] = affine.apply #[[MAP2]](%[[ARG7]])
// TILING-DAG:              %[[D9:.+]] = affine.min #[[MAP3]](%[[D8]])
// TILING:                %[[D10:.+]] = scf.for %[[ARG9:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// TILING-SAME:             iter_args(%[[ARG10:[a-zA-Z0-9_]+]] = %[[ARG8]]) -> (tensor<8x8x?x2x2x?xf32>) {
// TILING-DAG:                %[[D11:.+]] = affine.apply #[[MAP2]](%[[ARG9]])
// TILING-DAG:                %[[D12:.+]] = affine.min #[[MAP3]](%[[D11]])
// TILING:                  %[[D13:.+]] = scf.for %[[ARG11:[a-zA-Z0-9_]+]] = %[[C0]] to %[[D5]] step %[[C1]]
// TILING-SAME:               iter_args(%[[ARG12:[a-zA-Z0-9_]+]] = %[[ARG10]]) -> (tensor<8x8x?x2x2x?xf32>) {
// TILING:                    %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[EXTRACTED_SLICE]][%[[ARG5]], %[[D8]],
// TILING-SAME:                 %[[D11]], %[[ARG11]]] [1, %[[D9]], %[[D12]], 1] [1, 1, 1, 1] : tensor<?x10x10x?xf32> to
// TILING-SAME:                 tensor<?x?xf32>
// TILING:                    %[[EXTRACTED_SLICE_5:.+]] = tensor.extract_slice %[[ARG12]][0, 0, %[[ARG5]], %[[ARG7]],
// TILING-SAME:                 %[[ARG9]], %[[ARG11]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8x?x2x2x?xf32> to
// TILING-SAME:                 tensor<8x8xf32>
// TILING:                    %[[TILED_WINOGRAD:.+]] = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
// TILING-SAME:                                                  ins(%[[EXTRACTED_SLICE_3]] : tensor<?x?xf32>)
// TILING-SAME:                                                  outs(%[[EXTRACTED_SLICE_5]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// TILING:                    %[[INSERTED_SLICE_6:.+]] = tensor.insert_slice %[[TILED_WINOGRAD]] into %[[ARG12]][0, 0, %[[ARG5]],
// TILING-SAME:                 %[[ARG7]], %[[ARG9]], %[[ARG11]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8xf32>
// TILING-SAME:                 into tensor<8x8x?x2x2x?xf32>
// TILING:                    scf.yield %[[INSERTED_SLICE_6]] : tensor<8x8x?x2x2x?xf32>
// TILING:                  }
// TILING:                  scf.yield %[[D13]] : tensor<8x8x?x2x2x?xf32>
// TILING:                }
// TILING:                scf.yield %[[D10]] : tensor<8x8x?x2x2x?xf32>
// TILING:              }
// TILING:              scf.yield %[[D7]] : tensor<8x8x?x2x2x?xf32>
// TILING:            }
// TILING:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D6]] into %[[ARG4]][0, 0, %[[ARG1]], 0, 0,
// TILING-SAME:         %[[ARG3]]] [8, 8, %[[D3]], 2, 2, %[[D5]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x?x2x2x?xf32> into
// TILING-SAME:         tensor<8x8x1x2x2x1280xf32>
// TILING:            scf.yield %[[INSERTED_SLICE]] : tensor<8x8x1x2x2x1280xf32>
// TILING:          }
// TILING:          scf.yield %[[D4]] : tensor<8x8x1x2x2x1280xf32>
// TILING:        }
// TILING:        return %[[D2]] : tensor<8x8x1x2x2x1280xf32>
// TILING:      }

// DECOMP-DAG:  #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
// DECOMP-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// DECOMP-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 6)>
// DECOMP-DAG:  #[[MAP3:.+]] = affine_map<(d0) -> (-d0 + 10, 8)>
// DECOMP:      func.func @winograd_input_transform(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x10x10x1280xf32>) ->
// DECOMP-SAME:   tensor<8x8x1x2x2x1280xf32> {
// DECOMP:        %[[CST_1:.+]] = arith.constant 0.000000e+00 : f32
// DECOMP:        %[[D0:.+]] = tensor.empty() : tensor<8x8xf32>
// DECOMP:        %[[CST:.+]] = arith.constant dense<
// DECOMP:        %[[CST_0:.+]] = arith.constant dense<
// DECOMP:        %[[C0:.+]] = arith.constant 0 : index
// DECOMP:        %[[C1:.+]] = arith.constant 1 : index
// DECOMP:        %[[C2:.+]] = arith.constant 2 : index
// DECOMP:        %[[C1280:.+]] = arith.constant 1280 : index
// DECOMP:        %[[C32:.+]] = arith.constant 32 : index
// DECOMP:        %[[D1:.+]] = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
// DECOMP:        %[[D2:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1]] step %[[C1]]
// DECOMP-SAME:     iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D1]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// DECOMP-DAG:        %[[D3:.+]] = affine.min #[[MAP]](%[[ARG1]])[%[[C1]], %[[C1]]]
// DECOMP:          %[[D4:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1280]] step %[[C32]]
// DECOMP-SAME:       iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// DECOMP-DAG:          %[[D5:.+]] = affine.min #[[MAP1]](%[[ARG3]])[%[[C32]], %[[C1280]]]
// DECOMP:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG1]], 0, 0, %[[ARG3]]] [%[[D3]], 10,
// DECOMP-SAME:         10, %[[D5]]] [1, 1, 1, 1] : tensor<1x10x10x1280xf32> to tensor<?x10x10x?xf32>
// DECOMP:            %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[D1]][0, 0, %[[ARG1]], 0, 0, %[[ARG3]]] [8, 8,
// DECOMP-SAME:         %[[D3]], 2, 2, %[[D5]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x1280xf32> to
// DECOMP-SAME:         tensor<8x8x?x2x2x?xf32>
// DECOMP:            %[[D6:.+]] = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[D3]] step %[[C1]]
// DECOMP-SAME:         iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[EXTRACTED_SLICE_2]]) -> (tensor<8x8x?x2x2x?xf32>) {
// DECOMP:              %[[D7:.+]] = scf.for %[[ARG7:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// DECOMP-SAME:           iter_args(%[[ARG8:[a-zA-Z0-9_]+]] = %[[ARG6]]) -> (tensor<8x8x?x2x2x?xf32>) {
// DECOMP-DAG:              %[[D8:.+]] = affine.apply #[[MAP2]](%[[ARG7]])
// DECOMP-DAG:              %[[D9:.+]] = affine.min #[[MAP3]](%[[D8]])
// DECOMP:                %[[D10:.+]] = scf.for %[[ARG9:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// DECOMP-SAME:             iter_args(%[[ARG10:[a-zA-Z0-9_]+]] = %[[ARG8]]) -> (tensor<8x8x?x2x2x?xf32>) {
// DECOMP-DAG:                %[[D11:.+]] = affine.apply #[[MAP2]](%[[ARG9]])
// DECOMP-DAG:                %[[D12:.+]] = affine.min #[[MAP3]](%[[D11]])
// DECOMP:                  %[[D13:.+]] = scf.for %[[ARG11:[a-zA-Z0-9_]+]] = %[[C0]] to %[[D5]] step %[[C1]]
// DECOMP-SAME:               iter_args(%[[ARG12:[a-zA-Z0-9_]+]] = %[[ARG10]]) -> (tensor<8x8x?x2x2x?xf32>) {
// DECOMP:                    %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[EXTRACTED_SLICE]][%[[ARG5]], %[[D8]],
// DECOMP-SAME:                 %[[D11]], %[[ARG11]]] [1, %[[D9]], %[[D12]], 1] [1, 1, 1, 1] : tensor<?x10x10x?xf32> to
// DECOMP-SAME:                 tensor<?x?xf32>
// DECOMP:                    %[[EXTRACTED_SLICE_5:.+]] = tensor.extract_slice %[[ARG12]][0, 0, %[[ARG5]], %[[ARG7]],
// DECOMP-SAME:                 %[[ARG9]], %[[ARG11]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8x?x2x2x?xf32> to
// DECOMP-SAME:                 tensor<8x8xf32>
// DECOMP:                    %[[D14:.+]] = linalg.fill ins(%[[CST_1]] : f32) outs(%[[D0]] : tensor<8x8xf32>) ->
// DECOMP-SAME:                 tensor<8x8xf32>
// DECOMP:                    %[[INSERTED_SLICE_4:.+]] = tensor.insert_slice %[[EXTRACTED_SLICE_3]] into %[[D14]][0, 0]
// DECOMP-SAME:                 [%[[D9]], %[[D12]]] [1, 1] : tensor<?x?xf32> into tensor<8x8xf32>
// DECOMP:                    %[[D15:.+]] = linalg.fill ins(%[[CST_1]] : f32) outs(%[[EXTRACTED_SLICE_5]] :
// DECOMP-SAME:                 tensor<8x8xf32>) -> tensor<8x8xf32>
// DECOMP:                    %[[D16:.+]] = linalg.matmul ins(%[[INSERTED_SLICE_4]], %[[CST_0]] : tensor<8x8xf32>,
// DECOMP-SAME:                 tensor<8x8xf32>) outs(%[[D15]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// DECOMP:                    %[[D18:.+]] = linalg.matmul ins(%[[CST]], %[[D16]] : tensor<8x8xf32>, tensor<8x8xf32>)
// DECOMP-SAME:                 outs(%[[D15]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// DECOMP:                    %[[INSERTED_SLICE_6:.+]] = tensor.insert_slice %[[D18]] into %[[ARG12]][0, 0, %[[ARG5]],
// DECOMP-SAME:                 %[[ARG7]], %[[ARG9]], %[[ARG11]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8xf32>
// DECOMP-SAME:                 into tensor<8x8x?x2x2x?xf32>
// DECOMP:                    scf.yield %[[INSERTED_SLICE_6]] : tensor<8x8x?x2x2x?xf32>
// DECOMP:                  }
// DECOMP:                  scf.yield %[[D13]] : tensor<8x8x?x2x2x?xf32>
// DECOMP:                }
// DECOMP:                scf.yield %[[D10]] : tensor<8x8x?x2x2x?xf32>
// DECOMP:              }
// DECOMP:              scf.yield %[[D7]] : tensor<8x8x?x2x2x?xf32>
// DECOMP:            }
// DECOMP:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D6]] into %[[ARG4]][0, 0, %[[ARG1]], 0, 0,
// DECOMP-SAME:         %[[ARG3]]] [8, 8, %[[D3]], 2, 2, %[[D5]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x?x2x2x?xf32> into
// DECOMP-SAME:         tensor<8x8x1x2x2x1280xf32>
// DECOMP:            scf.yield %[[INSERTED_SLICE]] : tensor<8x8x1x2x2x1280xf32>
// DECOMP:          }
// DECOMP:          scf.yield %[[D4]] : tensor<8x8x1x2x2x1280xf32>
// DECOMP:        }
// DECOMP:        return %[[D2]] : tensor<8x8x1x2x2x1280xf32>
// DECOMP:      }

// CPU:        func.func @winograd_input_transform
// CPU-NOT:         iree_linalg_ext.winograd.input_transform

// GPU:        func.func @winograd_input_transform
// GPU:             iree_linalg_ext.winograd.input_transform

// SPIRV:      func.func @winograd_input_transform
// SPIRV-NOT:       iree_linalg_ext.winograd.input_transform

// -----

#map = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
#map1 = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
module {
  func.func @winograd_output_transform(%arg0: tensor<8x8x1x2x2x32xf32>) -> tensor<1x12x12x32xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %0 = tensor.empty() : tensor<1x12x12x32xf32>
    %1 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %0) -> (tensor<1x12x12x32xf32>) {
      %2 = affine.min #map(%arg1)[%c1, %c1]
      %3 = scf.for %arg3 = %c0 to %c32 step %c32 iter_args(%arg4 = %arg2) -> (tensor<1x12x12x32xf32>) {
        %4 = affine.min #map1(%arg3)[%c32, %c32]
        %extracted_slice = tensor.extract_slice %arg0[0, 0, %arg1, 0, 0, %arg3] [8, 8, %2, 2, 2, %4] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x32xf32> to tensor<8x8x?x2x2x?xf32>
        %extracted_slice_0 = tensor.extract_slice %0[%arg1, 0, 0, %arg3] [%2, 12, 12, %4] [1, 1, 1, 1] : tensor<1x12x12x32xf32> to tensor<?x12x12x?xf32>
        %5 = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2]) ins(%extracted_slice : tensor<8x8x?x2x2x?xf32>) outs(%extracted_slice_0 : tensor<?x12x12x?xf32>) -> tensor<?x12x12x?xf32>
        %inserted_slice = tensor.insert_slice %5 into %arg4[%arg1, 0, 0, %arg3] [%2, 12, 12, %4] [1, 1, 1, 1] : tensor<?x12x12x?xf32> into tensor<1x12x12x32xf32>
        scf.yield %inserted_slice : tensor<1x12x12x32xf32>
      }
      scf.yield %3 : tensor<1x12x12x32xf32>
    }
    return %1 : tensor<1x12x12x32xf32>
  }
}
// TILING-DAG:  #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
// TILING-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// TILING-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 6)>
// TILING:      func.func @winograd_output_transform(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<8x8x1x2x2x32xf32>) ->
// TILING-SAME:   tensor<1x12x12x32xf32> {
// TILING:        %[[C0:.+]] = arith.constant 0 : index
// TILING:        %[[C1:.+]] = arith.constant 1 : index
// TILING:        %[[C2:.+]] = arith.constant 2 : index
// TILING:        %[[C32:.+]] = arith.constant 32 : index
// TILING:        %[[D1:.+]] = tensor.empty() : tensor<1x12x12x32xf32>
// TILING:        %[[D2:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1]] step %[[C1]]
// TILING-SAME:     iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D1]]) -> (tensor<1x12x12x32xf32>) {
// TILING-DAG:        %[[D3:.+]] = affine.min #[[MAP]](%[[ARG1]])[%[[C1]], %[[C1]]]
// TILING:          %[[D4:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C32]] step %[[C32]]
// TILING-SAME:       iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<1x12x12x32xf32>) {
// TILING-DAG:          %[[D5:.+]] = affine.min #[[MAP1]](%[[ARG3]])[%[[C32]], %[[C32]]]
// TILING:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, 0, %[[ARG1]], 0, 0, %[[ARG3]]] [8, 8,
// TILING-SAME:         %[[D3]], 2, 2, %[[D5]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x32xf32> to tensor<8x8x?x2x2x?xf32>
// TILING:            %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[D1]][%[[ARG1]], 0, 0, %[[ARG3]]] [%[[D3]], 12,
// TILING-SAME:         12, %[[D5]]] [1, 1, 1, 1] : tensor<1x12x12x32xf32> to tensor<?x12x12x?xf32>
// TILING:            %[[D6:.+]] = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[D3]] step %[[C1]]
// TILING-SAME:         iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[EXTRACTED_SLICE_2]]) -> (tensor<?x12x12x?xf32>) {
// TILING:              %[[D7:.+]] = scf.for %[[ARG7:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// TILING-SAME:           iter_args(%[[ARG8:[a-zA-Z0-9_]+]] = %[[ARG6]]) -> (tensor<?x12x12x?xf32>) {
// TILING-DAG:              %[[D8:.+]] = affine.apply #[[MAP2]](%[[ARG7]])
// TILING:                %[[D9:.+]] = scf.for %[[ARG9:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// TILING-SAME:             iter_args(%[[ARG10:[a-zA-Z0-9_]+]] = %[[ARG8]]) -> (tensor<?x12x12x?xf32>) {
// TILING-DAG:                %[[D10:.+]] = affine.apply #[[MAP2]](%[[ARG9]])
// TILING:                  %[[D11:.+]] = scf.for %[[ARG11:[a-zA-Z0-9_]+]] = %[[C0]] to %[[D5]] step %[[C1]]
// TILING-SAME:               iter_args(%[[ARG12:[a-zA-Z0-9_]+]] = %[[ARG10]]) -> (tensor<?x12x12x?xf32>) {
// TILING:                    %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[EXTRACTED_SLICE]][0, 0, %[[ARG5]],
// TILING-SAME:                 %[[ARG7]], %[[ARG9]], %[[ARG11]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] :
// TILING-SAME:                 tensor<8x8x?x2x2x?xf32> to tensor<8x8xf32>
// TILING:                    %[[EXTRACTED_SLICE_4:.+]] = tensor.extract_slice %[[ARG12]][%[[ARG5]], %[[D8]], %[[D10]],
// TILING-SAME:                 %[[ARG11]]] [1, 6, 6, 1] [1, 1, 1, 1] : tensor<?x12x12x?xf32> to tensor<6x6xf32>
// TILING:                    %[[TILED_WINOGRAD:.+]] = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
// TILING-SAME:                                                  ins(%[[EXTRACTED_SLICE_3]] : tensor<8x8xf32>)
// TILING-SAME:                                                  outs(%[[EXTRACTED_SLICE_4]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// TILING:                    %[[INSERTED_SLICE_5:.+]] = tensor.insert_slice %[[TILED_WINOGRAD]] into %[[ARG12]][%[[ARG5]], %[[D8]],
// TILING-SAME:                 %[[D10]], %[[ARG11]]] [1, 6, 6, 1] [1, 1, 1, 1] : tensor<6x6xf32> into
// TILING-SAME:                 tensor<?x12x12x?xf32>
// TILING:                    scf.yield %[[INSERTED_SLICE_5]] : tensor<?x12x12x?xf32>
// TILING:                  }
// TILING:                  scf.yield %[[D11]] : tensor<?x12x12x?xf32>
// TILING:                }
// TILING:                scf.yield %[[D9]] : tensor<?x12x12x?xf32>
// TILING:              }
// TILING:              scf.yield %[[D7]] : tensor<?x12x12x?xf32>
// TILING:            }
// TILING:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D6]] into %[[ARG4]][%[[ARG1]], 0, 0, %[[ARG3]]]
// TILING-SAME:         [%[[D3]], 12, 12, %[[D5]]] [1, 1, 1, 1] : tensor<?x12x12x?xf32> into tensor<1x12x12x32xf32>
// TILING:            scf.yield %[[INSERTED_SLICE]] : tensor<1x12x12x32xf32>
// TILING:          }
// TILING:          scf.yield %[[D4]] : tensor<1x12x12x32xf32>
// TILING:        }
// TILING:        return %[[D2]] : tensor<1x12x12x32xf32>
// TILING:      }

// DECOMP-DAG:  #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
// DECOMP-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// DECOMP-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 6)>
// DECOMP:      func.func @winograd_output_transform(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<8x8x1x2x2x32xf32>) ->
// DECOMP-SAME:   tensor<1x12x12x32xf32> {
// DECOMP:        %[[CST:.+]] = arith.constant dense<
// DECOMP:        %[[CST_0:.+]] = arith.constant dense<
// DECOMP:        %[[CST_1:.+]] = arith.constant 0.000000e+00 : f32
// DECOMP:        %[[D0:.+]] = tensor.empty() : tensor<8x6xf32>
// DECOMP:        %[[C0:.+]] = arith.constant 0 : index
// DECOMP:        %[[C1:.+]] = arith.constant 1 : index
// DECOMP:        %[[C2:.+]] = arith.constant 2 : index
// DECOMP:        %[[C32:.+]] = arith.constant 32 : index
// DECOMP:        %[[D1:.+]] = tensor.empty() : tensor<1x12x12x32xf32>
// DECOMP:        %[[D2:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1]] step %[[C1]]
// DECOMP-SAME:     iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D1]]) -> (tensor<1x12x12x32xf32>) {
// DECOMP-DAG:        %[[D3:.+]] = affine.min #[[MAP]](%[[ARG1]])[%[[C1]], %[[C1]]]
// DECOMP:          %[[D4:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C32]] step %[[C32]]
// DECOMP-SAME:       iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<1x12x12x32xf32>) {
// DECOMP-DAG:          %[[D5:.+]] = affine.min #[[MAP1]](%[[ARG3]])[%[[C32]], %[[C32]]]
// DECOMP:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, 0, %[[ARG1]], 0, 0, %[[ARG3]]] [8, 8,
// DECOMP-SAME:         %[[D3]], 2, 2, %[[D5]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x32xf32> to tensor<8x8x?x2x2x?xf32>
// DECOMP:            %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[D1]][%[[ARG1]], 0, 0, %[[ARG3]]] [%[[D3]], 12,
// DECOMP-SAME:         12, %[[D5]]] [1, 1, 1, 1] : tensor<1x12x12x32xf32> to tensor<?x12x12x?xf32>
// DECOMP:            %[[D6:.+]] = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[D3]] step %[[C1]]
// DECOMP-SAME:         iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[EXTRACTED_SLICE_2]]) -> (tensor<?x12x12x?xf32>) {
// DECOMP:              %[[D7:.+]] = scf.for %[[ARG7:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// DECOMP-SAME:           iter_args(%[[ARG8:[a-zA-Z0-9_]+]] = %[[ARG6]]) -> (tensor<?x12x12x?xf32>) {
// DECOMP-DAG:              %[[D8:.+]] = affine.apply #[[MAP2]](%[[ARG7]])
// DECOMP:                %[[D9:.+]] = scf.for %[[ARG9:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// DECOMP-SAME:             iter_args(%[[ARG10:[a-zA-Z0-9_]+]] = %[[ARG8]]) -> (tensor<?x12x12x?xf32>) {
// DECOMP-DAG:                %[[D10:.+]] = affine.apply #[[MAP2]](%[[ARG9]])
// DECOMP:                  %[[D11:.+]] = scf.for %[[ARG11:[a-zA-Z0-9_]+]] = %[[C0]] to %[[D5]] step %[[C1]]
// DECOMP-SAME:               iter_args(%[[ARG12:[a-zA-Z0-9_]+]] = %[[ARG10]]) -> (tensor<?x12x12x?xf32>) {
// DECOMP:                    %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[EXTRACTED_SLICE]][0, 0, %[[ARG5]],
// DECOMP-SAME:                 %[[ARG7]], %[[ARG9]], %[[ARG11]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] :
// DECOMP-SAME:                 tensor<8x8x?x2x2x?xf32> to tensor<8x8xf32>
// DECOMP:                    %[[EXTRACTED_SLICE_4:.+]] = tensor.extract_slice %[[ARG12]][%[[ARG5]], %[[D8]], %[[D10]],
// DECOMP-SAME:                 %[[ARG11]]] [1, 6, 6, 1] [1, 1, 1, 1] : tensor<?x12x12x?xf32> to tensor<6x6xf32>
// DECOMP:                    %[[D12:.+]] = linalg.fill ins(%[[CST_1]] : f32) outs(%[[D0]] : tensor<8x6xf32>) ->
// DECOMP-SAME:                 tensor<8x6xf32>
// DECOMP:                    %[[D13:.+]] = linalg.matmul ins(%[[EXTRACTED_SLICE_3]], %[[CST_0]] : tensor<8x8xf32>,
// DECOMP-SAME:                 tensor<8x6xf32>) outs(%[[D12]] : tensor<8x6xf32>) -> tensor<8x6xf32>
// DECOMP:                    %[[D14:.+]] = linalg.fill ins(%[[CST_1]] : f32) outs(%[[EXTRACTED_SLICE_4]] :
// DECOMP-SAME:                 tensor<6x6xf32>) -> tensor<6x6xf32>
// DECOMP:                    %[[D15:.+]] = linalg.matmul ins(%[[CST]], %[[D13]] : tensor<6x8xf32>, tensor<8x6xf32>)
// DECOMP-SAME:                 outs(%[[D14]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// DECOMP:                    %[[INSERTED_SLICE_5:.+]] = tensor.insert_slice %[[D15]] into %[[ARG12]][%[[ARG5]], %[[D8]],
// DECOMP-SAME:                 %[[D10]], %[[ARG11]]] [1, 6, 6, 1] [1, 1, 1, 1] : tensor<6x6xf32> into
// DECOMP-SAME:                 tensor<?x12x12x?xf32>
// DECOMP:                    scf.yield %[[INSERTED_SLICE_5]] : tensor<?x12x12x?xf32>
// DECOMP:                  }
// DECOMP:                  scf.yield %[[D11]] : tensor<?x12x12x?xf32>
// DECOMP:                }
// DECOMP:                scf.yield %[[D9]] : tensor<?x12x12x?xf32>
// DECOMP:              }
// DECOMP:              scf.yield %[[D7]] : tensor<?x12x12x?xf32>
// DECOMP:            }
// DECOMP:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D6]] into %[[ARG4]][%[[ARG1]], 0, 0, %[[ARG3]]]
// DECOMP-SAME:         [%[[D3]], 12, 12, %[[D5]]] [1, 1, 1, 1] : tensor<?x12x12x?xf32> into tensor<1x12x12x32xf32>
// DECOMP:            scf.yield %[[INSERTED_SLICE]] : tensor<1x12x12x32xf32>
// DECOMP:          }
// DECOMP:          scf.yield %[[D4]] : tensor<1x12x12x32xf32>
// DECOMP:        }
// DECOMP:        return %[[D2]] : tensor<1x12x12x32xf32>
// DECOMP:      }

// CPU:        func.func @winograd_output_transform
// CPU-NOT:         iree_linalg_ext.winograd.output_transform

// GPU:        func.func @winograd_output_transform
// GPU:             iree_linalg_ext.winograd.output_transform

// SPIRV:      func.func @winograd_output_transform
// SPIRV-NOT:       iree_linalg_ext.winograd.output_transform

// -----

#map = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
#map1 = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
module {
  func.func @winograd_input_transform_nchw(%arg0: tensor<1x1280x10x10xf32>) -> tensor<8x8x1x2x2x1280xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1280 = arith.constant 1280 : index
    %c32 = arith.constant 32 : index
    %0 = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
    %1 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %0) -> (tensor<8x8x1x2x2x1280xf32>) {
      %2 = affine.min #map(%arg1)[%c1, %c1]
      %3 = scf.for %arg3 = %c0 to %c1280 step %c32 iter_args(%arg4 = %arg2) -> (tensor<8x8x1x2x2x1280xf32>) {
        %4 = affine.min #map1(%arg3)[%c32, %c1280]
        %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3, 0, 0] [%2, %4, 10, 10] [1, 1, 1, 1] : tensor<1x1280x10x10xf32> to tensor<?x?x10x10xf32>
        %extracted_slice_0 = tensor.extract_slice %0[0, 0, %arg1, 0, 0, %arg3] [8, 8, %2, 2, 2, %4] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x1280xf32> to tensor<8x8x?x2x2x?xf32>
        %5 = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([2, 3]) ins(%extracted_slice : tensor<?x?x10x10xf32>) outs(%extracted_slice_0 : tensor<8x8x?x2x2x?xf32>) -> tensor<8x8x?x2x2x?xf32>
        %inserted_slice = tensor.insert_slice %5 into %arg4[0, 0, %arg1, 0, 0, %arg3] [8, 8, %2, 2, 2, %4] [1, 1, 1, 1, 1, 1] : tensor<8x8x?x2x2x?xf32> into tensor<8x8x1x2x2x1280xf32>
        scf.yield %inserted_slice : tensor<8x8x1x2x2x1280xf32>
      }
      scf.yield %3 : tensor<8x8x1x2x2x1280xf32>
    }
    return %1 : tensor<8x8x1x2x2x1280xf32>
  }
}
// TILING-DAG:  #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
// TILING-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// TILING-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 6)>
// TILING-DAG:  #[[MAP3:.+]] = affine_map<(d0) -> (-d0 + 10, 8)>
// TILING:      func.func @winograd_input_transform_nchw(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x1280x10x10xf32>) ->
// TILING-SAME:   tensor<8x8x1x2x2x1280xf32> {
// TILING:        %[[C0:.+]] = arith.constant 0 : index
// TILING:        %[[C1:.+]] = arith.constant 1 : index
// TILING:        %[[C2:.+]] = arith.constant 2 : index
// TILING:        %[[C1280:.+]] = arith.constant 1280 : index
// TILING:        %[[C32:.+]] = arith.constant 32 : index
// TILING:        %[[D1:.+]] = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
// TILING:        %[[D2:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1]] step %[[C1]]
// TILING-SAME:     iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D1]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// TILING-DAG:        %[[D3:.+]] = affine.min #[[MAP]](%[[ARG1]])[%[[C1]], %[[C1]]]
// TILING:          %[[D4:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1280]] step %[[C32]]
// TILING-SAME:       iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// TILING-DAG:          %[[D5:.+]] = affine.min #[[MAP1]](%[[ARG3]])[%[[C32]], %[[C1280]]]
// TILING:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG1]], %[[ARG3]], 0, 0] [%[[D3]],
// TILING-SAME:         %[[D5]], 10, 10] [1, 1, 1, 1] : tensor<1x1280x10x10xf32> to tensor<?x?x10x10xf32>
// TILING:            %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[D1]][0, 0, %[[ARG1]], 0, 0, %[[ARG3]]] [8, 8,
// TILING-SAME:         %[[D3]], 2, 2, %[[D5]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x1280xf32> to
// TILING-SAME:         tensor<8x8x?x2x2x?xf32>
// TILING:            %[[D6:.+]] = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[D3]] step %[[C1]]
// TILING-SAME:         iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[EXTRACTED_SLICE_2]]) -> (tensor<8x8x?x2x2x?xf32>) {
// TILING:              %[[D7:.+]] = scf.for %[[ARG7:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// TILING-SAME:           iter_args(%[[ARG8:[a-zA-Z0-9_]+]] = %[[ARG6]]) -> (tensor<8x8x?x2x2x?xf32>) {
// TILING-DAG:              %[[D8:.+]] = affine.apply #[[MAP2]](%[[ARG7]])
// TILING-DAG:              %[[D9:.+]] = affine.min #[[MAP3]](%[[D8]])
// TILING:                %[[D10:.+]] = scf.for %[[ARG9:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// TILING-SAME:             iter_args(%[[ARG10:[a-zA-Z0-9_]+]] = %[[ARG8]]) -> (tensor<8x8x?x2x2x?xf32>) {
// TILING-DAG:                %[[D11:.+]] = affine.apply #[[MAP2]](%[[ARG9]])
// TILING-DAG:                %[[D12:.+]] = affine.min #[[MAP3]](%[[D11]])
// TILING:                  %[[D13:.+]] = scf.for %[[ARG11:[a-zA-Z0-9_]+]] = %[[C0]] to %[[D5]] step %[[C1]]
// TILING-SAME:               iter_args(%[[ARG12:[a-zA-Z0-9_]+]] = %[[ARG10]]) -> (tensor<8x8x?x2x2x?xf32>) {
// TILING:                    %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[EXTRACTED_SLICE]][%[[ARG5]],
// TILING-SAME:                 %[[ARG11]], %[[D8]], %[[D11]]] [1, 1, %[[D9]], %[[D12]]] [1, 1, 1, 1] :
// TILING-SAME:                 tensor<?x?x10x10xf32> to tensor<?x?xf32>
// TILING:                    %[[EXTRACTED_SLICE_5:.+]] = tensor.extract_slice %[[ARG12]][0, 0, %[[ARG5]], %[[ARG7]],
// TILING-SAME:                 %[[ARG9]], %[[ARG11]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8x?x2x2x?xf32> to
// TILING-SAME:                 tensor<8x8xf32>
// TILING:                    %[[TILED_WINOGRAD:.+]] = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([2, 3])
// TILING-SAME:                                                  ins(%[[EXTRACTED_SLICE_3]] : tensor<?x?xf32>)
// TILING-SAME:                                                  outs(%[[EXTRACTED_SLICE_5]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// TILING:                    %[[INSERTED_SLICE_6:.+]] = tensor.insert_slice %[[TILED_WINOGRAD]] into %[[ARG12]][0, 0, %[[ARG5]],
// TILING-SAME:                 %[[ARG7]], %[[ARG9]], %[[ARG11]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8xf32>
// TILING-SAME:                 into tensor<8x8x?x2x2x?xf32>
// TILING:                    scf.yield %[[INSERTED_SLICE_6]] : tensor<8x8x?x2x2x?xf32>
// TILING:                  }
// TILING:                  scf.yield %[[D13]] : tensor<8x8x?x2x2x?xf32>
// TILING:                }
// TILING:                scf.yield %[[D10]] : tensor<8x8x?x2x2x?xf32>
// TILING:              }
// TILING:              scf.yield %[[D7]] : tensor<8x8x?x2x2x?xf32>
// TILING:            }
// TILING:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D6]] into %[[ARG4]][0, 0, %[[ARG1]], 0, 0,
// TILING-SAME:         %[[ARG3]]] [8, 8, %[[D3]], 2, 2, %[[D5]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x?x2x2x?xf32> into
// TILING-SAME:         tensor<8x8x1x2x2x1280xf32>
// TILING:            scf.yield %[[INSERTED_SLICE]] : tensor<8x8x1x2x2x1280xf32>
// TILING:          }
// TILING:          scf.yield %[[D4]] : tensor<8x8x1x2x2x1280xf32>
// TILING:        }
// TILING:        return %[[D2]] : tensor<8x8x1x2x2x1280xf32>
// TILING:      }
// TILING:    }

// DECOMP-DAG:  #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
// DECOMP-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// DECOMP-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 6)>
// DECOMP-DAG:  #[[MAP3:.+]] = affine_map<(d0) -> (-d0 + 10, 8)>
// DECOMP:      func.func @winograd_input_transform_nchw(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x1280x10x10xf32>) ->
// DECOMP-SAME:   tensor<8x8x1x2x2x1280xf32> {
// DECOMP:        %[[CST_1:.+]] = arith.constant 0.000000e+00 : f32
// DECOMP:        %[[D0:.+]] = tensor.empty() : tensor<8x8xf32>
// DECOMP:        %[[CST:.+]] = arith.constant dense<
// DECOMP:        %[[CST_0:.+]] = arith.constant dense<
// DECOMP:        %[[C0:.+]] = arith.constant 0 : index
// DECOMP:        %[[C1:.+]] = arith.constant 1 : index
// DECOMP:        %[[C2:.+]] = arith.constant 2 : index
// DECOMP:        %[[C1280:.+]] = arith.constant 1280 : index
// DECOMP:        %[[C32:.+]] = arith.constant 32 : index
// DECOMP:        %[[D1:.+]] = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
// DECOMP:        %[[D2:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1]] step %[[C1]]
// DECOMP-SAME:     iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D1]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// DECOMP-DAG:        %[[D3:.+]] = affine.min #[[MAP]](%[[ARG1]])[%[[C1]], %[[C1]]]
// DECOMP:          %[[D4:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1280]] step %[[C32]]
// DECOMP-SAME:       iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// DECOMP-DAG:          %[[D5:.+]] = affine.min #[[MAP1]](%[[ARG3]])[%[[C32]], %[[C1280]]]
// DECOMP:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG1]], %[[ARG3]], 0, 0] [%[[D3]],
// DECOMP-SAME:         %[[D5]], 10, 10] [1, 1, 1, 1] : tensor<1x1280x10x10xf32> to tensor<?x?x10x10xf32>
// DECOMP:            %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[D1]][0, 0, %[[ARG1]], 0, 0, %[[ARG3]]] [8, 8,
// DECOMP-SAME:         %[[D3]], 2, 2, %[[D5]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x1280xf32> to
// DECOMP-SAME:         tensor<8x8x?x2x2x?xf32>
// DECOMP:            %[[D6:.+]] = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[D3]] step %[[C1]]
// DECOMP-SAME:         iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[EXTRACTED_SLICE_2]]) -> (tensor<8x8x?x2x2x?xf32>) {
// DECOMP:              %[[D7:.+]] = scf.for %[[ARG7:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// DECOMP-SAME:           iter_args(%[[ARG8:[a-zA-Z0-9_]+]] = %[[ARG6]]) -> (tensor<8x8x?x2x2x?xf32>) {
// DECOMP-DAG:              %[[D8:.+]] = affine.apply #[[MAP2]](%[[ARG7]])
// DECOMP-DAG:              %[[D9:.+]] = affine.min #[[MAP3]](%[[D8]])
// DECOMP:                %[[D10:.+]] = scf.for %[[ARG9:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// DECOMP-SAME:             iter_args(%[[ARG10:[a-zA-Z0-9_]+]] = %[[ARG8]]) -> (tensor<8x8x?x2x2x?xf32>) {
// DECOMP-DAG:                %[[D11:.+]] = affine.apply #[[MAP2]](%[[ARG9]])
// DECOMP-DAG:                %[[D12:.+]] = affine.min #[[MAP3]](%[[D11]])
// DECOMP:                  %[[D13:.+]] = scf.for %[[ARG11:[a-zA-Z0-9_]+]] = %[[C0]] to %[[D5]] step %[[C1]]
// DECOMP-SAME:               iter_args(%[[ARG12:[a-zA-Z0-9_]+]] = %[[ARG10]]) -> (tensor<8x8x?x2x2x?xf32>) {
// DECOMP:                    %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[EXTRACTED_SLICE]][%[[ARG5]],
// DECOMP-SAME:                 %[[ARG11]], %[[D8]], %[[D11]]] [1, 1, %[[D9]], %[[D12]]] [1, 1, 1, 1] :
// DECOMP-SAME:                 tensor<?x?x10x10xf32> to tensor<?x?xf32>
// DECOMP:                    %[[EXTRACTED_SLICE_5:.+]] = tensor.extract_slice %[[ARG12]][0, 0, %[[ARG5]], %[[ARG7]],
// DECOMP-SAME:                 %[[ARG9]], %[[ARG11]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8x?x2x2x?xf32> to
// DECOMP-SAME:                 tensor<8x8xf32>
// DECOMP:                    %[[D14:.+]] = linalg.fill ins(%[[CST_1]] : f32) outs(%[[D0]] : tensor<8x8xf32>) ->
// DECOMP-SAME:                 tensor<8x8xf32>
// DECOMP:                    %[[INSERTED_SLICE_4:.+]] = tensor.insert_slice %[[EXTRACTED_SLICE_3]] into %[[D14]][0, 0]
// DECOMP-SAME:                 [%[[D9]], %[[D12]]] [1, 1] : tensor<?x?xf32> into tensor<8x8xf32>
// DECOMP:                    %[[D15:.+]] = linalg.fill ins(%[[CST_1]] : f32) outs(%[[EXTRACTED_SLICE_5]] :
// DECOMP-SAME:                 tensor<8x8xf32>) -> tensor<8x8xf32>
// DECOMP:                    %[[D16:.+]] = linalg.matmul ins(%[[INSERTED_SLICE_4]], %[[CST_0]] : tensor<8x8xf32>,
// DECOMP-SAME:                 tensor<8x8xf32>) outs(%[[D15]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// DECOMP:                    %[[D18:.+]] = linalg.matmul ins(%[[CST]], %[[D16]] : tensor<8x8xf32>, tensor<8x8xf32>)
// DECOMP-SAME:                 outs(%[[D15]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// DECOMP:                    %[[INSERTED_SLICE_6:.+]] = tensor.insert_slice %[[D18]] into %[[ARG12]][0, 0, %[[ARG5]],
// DECOMP-SAME:                 %[[ARG7]], %[[ARG9]], %[[ARG11]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8xf32>
// DECOMP-SAME:                 into tensor<8x8x?x2x2x?xf32>
// DECOMP:                    scf.yield %[[INSERTED_SLICE_6]] : tensor<8x8x?x2x2x?xf32>
// DECOMP:                  }
// DECOMP:                  scf.yield %[[D13]] : tensor<8x8x?x2x2x?xf32>
// DECOMP:                }
// DECOMP:                scf.yield %[[D10]] : tensor<8x8x?x2x2x?xf32>
// DECOMP:              }
// DECOMP:              scf.yield %[[D7]] : tensor<8x8x?x2x2x?xf32>
// DECOMP:            }
// DECOMP:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D6]] into %[[ARG4]][0, 0, %[[ARG1]], 0, 0,
// DECOMP-SAME:         %[[ARG3]]] [8, 8, %[[D3]], 2, 2, %[[D5]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x?x2x2x?xf32> into
// DECOMP-SAME:         tensor<8x8x1x2x2x1280xf32>
// DECOMP:            scf.yield %[[INSERTED_SLICE]] : tensor<8x8x1x2x2x1280xf32>
// DECOMP:          }
// DECOMP:          scf.yield %[[D4]] : tensor<8x8x1x2x2x1280xf32>
// DECOMP:        }
// DECOMP:        return %[[D2]] : tensor<8x8x1x2x2x1280xf32>
// DECOMP:      }
// DECOMP:    }

// CPU:        func.func @winograd_input_transform_nchw
// CPU-NOT:         iree_linalg_ext.winograd.input_transform

// GPU:        func.func @winograd_input_transform_nchw
// GPU:             iree_linalg_ext.winograd.input_transform

// SPIRV:      func.func @winograd_input_transform_nchw
// SPIRV-NOT:       iree_linalg_ext.winograd.input_transform

// -----

#map = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
#map1 = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
module {
  func.func @winograd_output_transform_nchw(%arg0: tensor<8x8x1x2x2x32xf32>) -> tensor<1x32x12x12xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %0 = tensor.empty() : tensor<1x32x12x12xf32>
    %1 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %0) -> (tensor<1x32x12x12xf32>) {
      %2 = affine.min #map(%arg1)[%c1, %c1]
      %3 = scf.for %arg3 = %c0 to %c32 step %c32 iter_args(%arg4 = %arg2) -> (tensor<1x32x12x12xf32>) {
        %4 = affine.min #map1(%arg3)[%c32, %c32]
        %extracted_slice = tensor.extract_slice %arg0[0, 0, %arg1, 0, 0, %arg3] [8, 8, %2, 2, 2, %4] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x32xf32> to tensor<8x8x?x2x2x?xf32>
        %extracted_slice_0 = tensor.extract_slice %0[%arg1, %arg3, 0, 0] [%2, %4, 12, 12] [1, 1, 1, 1] : tensor<1x32x12x12xf32> to tensor<?x?x12x12xf32>
        %5 = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([2, 3]) ins(%extracted_slice : tensor<8x8x?x2x2x?xf32>) outs(%extracted_slice_0 : tensor<?x?x12x12xf32>) -> tensor<?x?x12x12xf32>
        %inserted_slice = tensor.insert_slice %5 into %arg4[%arg1, %arg3, 0, 0] [%2, %4, 12, 12] [1, 1, 1, 1] : tensor<?x?x12x12xf32> into tensor<1x32x12x12xf32>
        scf.yield %inserted_slice : tensor<1x32x12x12xf32>
      }
      scf.yield %3 : tensor<1x32x12x12xf32>
    }
    return %1 : tensor<1x32x12x12xf32>
  }
}
// TILING-DAG:  #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
// TILING-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// TILING-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 6)>
// TILING:      func.func @winograd_output_transform_nchw(%[[ARG0:.+]]: tensor<8x8x1x2x2x32xf32>) -> tensor<1x32x12x12xf32>
// TILING-SAME:   {
// TILING:        %[[C0:.+]] = arith.constant 0 : index
// TILING:        %[[C1:.+]] = arith.constant 1 : index
// TILING:        %[[C2:.+]] = arith.constant 2 : index
// TILING:        %[[C32:.+]] = arith.constant 32 : index
// TILING:        %[[D1:.+]] = tensor.empty() : tensor<1x32x12x12xf32>
// TILING:        %[[D2:.+]] = scf.for %[[ARG1:.+]] = %[[C0]] to %[[C1]] step %[[C1]] iter_args(%[[ARG2:.+]] = %[[D1]]) ->
// TILING-SAME:     (tensor<1x32x12x12xf32>) {
// TILING-DAG:        %[[D3:.+]] = affine.min #[[MAP]](%[[ARG1]])[%[[C1]], %[[C1]]]
// TILING:          %[[D4:.+]] = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C32]] step %[[C32]] iter_args(%[[ARG4:.+]] = %[[ARG2]]) ->
// TILING-SAME:       (tensor<1x32x12x12xf32>) {
// TILING-DAG:          %[[D5:.+]] = affine.min #[[MAP1]](%[[ARG3]])[%[[C32]], %[[C32]]]
// TILING:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, 0, %[[ARG1]], 0, 0, %[[ARG3]]] [8, 8,
// TILING-SAME:         %[[D3]], 2, 2, %[[D5]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x32xf32> to tensor<8x8x?x2x2x?xf32>
// TILING:            %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[D1]][%[[ARG1]], %[[ARG3]], 0, 0] [%[[D3]],
// TILING-SAME:         %[[D5]], 12, 12] [1, 1, 1, 1] : tensor<1x32x12x12xf32> to tensor<?x?x12x12xf32>
// TILING:            %[[D6:.+]] = scf.for %[[ARG5:.+]] = %[[C0]] to %[[D3]] step %[[C1]] iter_args(%[[ARG6:.+]] =
// TILING-SAME:         %[[EXTRACTED_SLICE_2]]) -> (tensor<?x?x12x12xf32>) {
// TILING:              %[[D7:.+]] = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG8:.+]] = %[[ARG6]]) ->
// TILING-SAME:           (tensor<?x?x12x12xf32>) {
// TILING-DAG:              %[[D8:.+]] = affine.apply #[[MAP2]](%[[ARG7]])
// TILING:                %[[D9:.+]] = scf.for %[[ARG9:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG10:.+]] = %[[ARG8]])
// TILING-SAME:             -> (tensor<?x?x12x12xf32>) {
// TILING-DAG:                %[[D10:.+]] = affine.apply #[[MAP2]](%[[ARG9]])
// TILING:                  %[[D11:.+]] = scf.for %[[ARG11:.+]] = %[[C0]] to %[[D5]] step %[[C1]] iter_args(%[[ARG12:.+]] =
// TILING-SAME:               %[[ARG10]]) -> (tensor<?x?x12x12xf32>) {
// TILING:                    %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[EXTRACTED_SLICE]][0, 0, %[[ARG5]],
// TILING-SAME:                 %[[ARG7]], %[[ARG9]], %[[ARG11]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] :
// TILING-SAME:                 tensor<8x8x?x2x2x?xf32> to tensor<8x8xf32>
// TILING:                    %[[EXTRACTED_SLICE_4:.+]] = tensor.extract_slice %[[ARG12]][%[[ARG5]], %[[ARG11]], %[[D8]],
// TILING-SAME:                 %[[D10]]] [1, 1, 6, 6] [1, 1, 1, 1] : tensor<?x?x12x12xf32> to tensor<6x6xf32>
// TILING:                    %[[TILED_WINOGRAD:.+]] = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([2, 3])
// TILING-SAME:                                                  ins(%[[EXTRACTED_SLICE_3]] : tensor<8x8xf32>)
// TILING-SAME:                                                  outs(%[[EXTRACTED_SLICE_4]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// TILING:                    %[[INSERTED_SLICE_5:.+]] = tensor.insert_slice %[[TILED_WINOGRAD]] into %[[ARG12]][%[[ARG5]],
// TILING-SAME:                 %[[ARG11]], %[[D8]], %[[D10]]] [1, 1, 6, 6] [1, 1, 1, 1] : tensor<6x6xf32> into
// TILING-SAME:                 tensor<?x?x12x12xf32>
// TILING:                    scf.yield %[[INSERTED_SLICE_5]] : tensor<?x?x12x12xf32>
// TILING:                  }
// TILING:                  scf.yield %[[D11]] : tensor<?x?x12x12xf32>
// TILING:                }
// TILING:                scf.yield %[[D9]] : tensor<?x?x12x12xf32>
// TILING:              }
// TILING:              scf.yield %[[D7]] : tensor<?x?x12x12xf32>
// TILING:            }
// TILING:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D6]] into %[[ARG4]][%[[ARG1]], %[[ARG3]], 0, 0]
// TILING-SAME:         [%[[D3]], %[[D5]], 12, 12] [1, 1, 1, 1] : tensor<?x?x12x12xf32> into tensor<1x32x12x12xf32>
// TILING:            scf.yield %[[INSERTED_SLICE]] : tensor<1x32x12x12xf32>
// TILING:          }
// TILING:          scf.yield %[[D4]] : tensor<1x32x12x12xf32>
// TILING:        }
// TILING:        return %[[D2]] : tensor<1x32x12x12xf32>
// TILING:      }

// DECOMP-DAG:  #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
// DECOMP-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// DECOMP-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 6)>
// DECOMP:      func.func @winograd_output_transform_nchw(%[[ARG0:.+]]: tensor<8x8x1x2x2x32xf32>) -> tensor<1x32x12x12xf32>
// DECOMP-SAME:   {
// DECOMP:        %[[CST:.+]] = arith.constant dense<
// DECOMP:        %[[CST_0:.+]] = arith.constant dense<
// DECOMP:        %[[CST_1:.+]] = arith.constant 0.000000e+00 : f32
// DECOMP:        %[[D0:.+]] = tensor.empty() : tensor<8x6xf32>
// DECOMP:        %[[C0:.+]] = arith.constant 0 : index
// DECOMP:        %[[C1:.+]] = arith.constant 1 : index
// DECOMP:        %[[C2:.+]] = arith.constant 2 : index
// DECOMP:        %[[C32:.+]] = arith.constant 32 : index
// DECOMP:        %[[D1:.+]] = tensor.empty() : tensor<1x32x12x12xf32>
// DECOMP:        %[[D2:.+]] = scf.for %[[ARG1:.+]] = %[[C0]] to %[[C1]] step %[[C1]] iter_args(%[[ARG2:.+]] = %[[D1]]) ->
// DECOMP-SAME:     (tensor<1x32x12x12xf32>) {
// DECOMP-DAG:        %[[D3:.+]] = affine.min #[[MAP]](%[[ARG1]])[%[[C1]], %[[C1]]]
// DECOMP:          %[[D4:.+]] = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C32]] step %[[C32]] iter_args(%[[ARG4:.+]] = %[[ARG2]]) ->
// DECOMP-SAME:       (tensor<1x32x12x12xf32>) {
// DECOMP-DAG:          %[[D5:.+]] = affine.min #[[MAP1]](%[[ARG3]])[%[[C32]], %[[C32]]]
// DECOMP:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, 0, %[[ARG1]], 0, 0, %[[ARG3]]] [8, 8,
// DECOMP-SAME:         %[[D3]], 2, 2, %[[D5]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x32xf32> to tensor<8x8x?x2x2x?xf32>
// DECOMP:            %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[D1]][%[[ARG1]], %[[ARG3]], 0, 0] [%[[D3]],
// DECOMP-SAME:         %[[D5]], 12, 12] [1, 1, 1, 1] : tensor<1x32x12x12xf32> to tensor<?x?x12x12xf32>
// DECOMP:            %[[D6:.+]] = scf.for %[[ARG5:.+]] = %[[C0]] to %[[D3]] step %[[C1]] iter_args(%[[ARG6:.+]] =
// DECOMP-SAME:         %[[EXTRACTED_SLICE_2]]) -> (tensor<?x?x12x12xf32>) {
// DECOMP:              %[[D7:.+]] = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG8:.+]] = %[[ARG6]]) ->
// DECOMP-SAME:           (tensor<?x?x12x12xf32>) {
// DECOMP-DAG:              %[[D8:.+]] = affine.apply #[[MAP2]](%[[ARG7]])
// DECOMP:                %[[D9:.+]] = scf.for %[[ARG9:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG10:.+]] = %[[ARG8]])
// DECOMP-SAME:             -> (tensor<?x?x12x12xf32>) {
// DECOMP-DAG:                %[[D10:.+]] = affine.apply #[[MAP2]](%[[ARG9]])
// DECOMP:                  %[[D11:.+]] = scf.for %[[ARG11:.+]] = %[[C0]] to %[[D5]] step %[[C1]] iter_args(%[[ARG12:.+]] =
// DECOMP-SAME:               %[[ARG10]]) -> (tensor<?x?x12x12xf32>) {
// DECOMP:                    %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[EXTRACTED_SLICE]][0, 0, %[[ARG5]],
// DECOMP-SAME:                 %[[ARG7]], %[[ARG9]], %[[ARG11]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] :
// DECOMP-SAME:                 tensor<8x8x?x2x2x?xf32> to tensor<8x8xf32>
// DECOMP:                    %[[EXTRACTED_SLICE_4:.+]] = tensor.extract_slice %[[ARG12]][%[[ARG5]], %[[ARG11]], %[[D8]],
// DECOMP-SAME:                 %[[D10]]] [1, 1, 6, 6] [1, 1, 1, 1] : tensor<?x?x12x12xf32> to tensor<6x6xf32>
// DECOMP:                    %[[D12:.+]] = linalg.fill ins(%[[CST_1]] : f32) outs(%[[D0]] : tensor<8x6xf32>) ->
// DECOMP-SAME:                 tensor<8x6xf32>
// DECOMP:                    %[[D13:.+]] = linalg.matmul ins(%[[EXTRACTED_SLICE_3]], %[[CST_0]] : tensor<8x8xf32>,
// DECOMP-SAME:                 tensor<8x6xf32>) outs(%[[D12]] : tensor<8x6xf32>) -> tensor<8x6xf32>
// DECOMP:                    %[[D14:.+]] = linalg.fill ins(%[[CST_1]] : f32) outs(%[[EXTRACTED_SLICE_4]] : tensor<6x6xf32>)
// DECOMP-SAME:                 -> tensor<6x6xf32>
// DECOMP:                    %[[D15:.+]] = linalg.matmul ins(%[[CST]], %[[D13]] : tensor<6x8xf32>, tensor<8x6xf32>)
// DECOMP-SAME:                 outs(%[[D14]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// DECOMP:                    %[[INSERTED_SLICE_5:.+]] = tensor.insert_slice %[[D15]] into %[[ARG12]][%[[ARG5]],
// DECOMP-SAME:                 %[[ARG11]], %[[D8]], %[[D10]]] [1, 1, 6, 6] [1, 1, 1, 1] : tensor<6x6xf32> into
// DECOMP-SAME:                 tensor<?x?x12x12xf32>
// DECOMP:                    scf.yield %[[INSERTED_SLICE_5]] : tensor<?x?x12x12xf32>
// DECOMP:                  }
// DECOMP:                  scf.yield %[[D11]] : tensor<?x?x12x12xf32>
// DECOMP:                }
// DECOMP:                scf.yield %[[D9]] : tensor<?x?x12x12xf32>
// DECOMP:              }
// DECOMP:              scf.yield %[[D7]] : tensor<?x?x12x12xf32>
// DECOMP:            }
// DECOMP:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D6]] into %[[ARG4]][%[[ARG1]], %[[ARG3]], 0, 0]
// DECOMP-SAME:         [%[[D3]], %[[D5]], 12, 12] [1, 1, 1, 1] : tensor<?x?x12x12xf32> into tensor<1x32x12x12xf32>
// DECOMP:            scf.yield %[[INSERTED_SLICE]] : tensor<1x32x12x12xf32>
// DECOMP:          }
// DECOMP:          scf.yield %[[D4]] : tensor<1x32x12x12xf32>
// DECOMP:        }
// DECOMP:        return %[[D2]] : tensor<1x32x12x12xf32>
// DECOMP:      }

// CPU:        func.func @winograd_output_transform_nchw
// CPU-NOT:         iree_linalg_ext.winograd.output_transform

// GPU:        func.func @winograd_output_transform_nchw
// GPU:             iree_linalg_ext.winograd.output_transform

// SPIRV:      func.func @winograd_output_transform_nchw
// SPIRV-NOT:       iree_linalg_ext.winograd.output_transform
