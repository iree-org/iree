// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-linalg-ext-tile-and-decompose-attention{onlyTile}),cse)" %s | FileCheck %s --check-prefix=TILING

func.func @attention(%query: tensor<1x1024x64xf32>, %key: tensor<1x1024x64xf32>, %value: tensor<1x1024x64xf32>) -> tensor<1x1024x64xf32> {
  %0 = tensor.empty() : tensor<1x1024x64xf32>
  %scale = arith.constant 0.05 : f32
  %1 = iree_linalg_ext.attention ins(%query, %key, %value, %scale : tensor<1x1024x64xf32>, tensor<1x1024x64xf32>, tensor<1x1024x64xf32>, f32) outs(%0 : tensor<1x1024x64xf32>) -> tensor<1x1024x64xf32>
  return %1 : tensor<1x1024x64xf32>
}

// TILING-DAG:  #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// TILING-DAG:  #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
// TILING-LABEL:      func.func @attention
// TILING-SAME: (%[[QUERY:.+]]: tensor<1x1024x64xf32>, %[[KEY:.+]]: tensor<1x1024x64xf32>, %[[VALUE:.+]]: tensor<1x1024x64xf32>)
// TILING-DAG:    %[[D0:.+]] = tensor.empty() : tensor<1x1024x64xf32>
// TILING-DAG:    %[[D1:.+]] = tensor.empty() : tensor<1024x64xf32>
// TILING-DAG:    %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// TILING-DAG:    %[[CST_1:.+]] = arith.constant 5.000000e-02 : f32
// TILING:        %[[D2:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D1]] : tensor<1024x64xf32>)
// TILING-DAG:    %[[CST_0:.+]] = arith.constant -1.000000e+30 : f32
// TILING:        %[[D3:.+]] = tensor.empty() : tensor<1024xf32>
// TILING:        %[[D4:.+]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[D3]] : tensor<1024xf32>)
// TILING:        %[[D5:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D3]] : tensor<1024xf32>)
// TILING-DAG:    %[[C0:.+]] = arith.constant 0 : index
// TILING-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
// TILING:        %[[D6:.+]]:3 = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C1024]] step %[[C1024]]
// TILING-SAME:     iter_args(%[[ARG4:.+]] = %[[D2]], %[[ARG5:.+]] = %[[D4]], %[[ARG6:.+]] = %[[D5]])
// TILING:          %[[K_S:.+]] = tensor.extract_slice %[[KEY]][0, %[[ARG3]], 0] [1, 1024, 64] [1, 1, 1] :
// TILING:          %[[V_S:.+]] = tensor.extract_slice %[[VALUE]][0, %[[ARG3]], 0] [1, 1024, 64] [1, 1, 1] :
// TILING:          %[[Q_S:.+]] = tensor.extract_slice %[[QUERY]][0, 0, 0] [1, 1024, 64] [1, 1, 1] :
// TILING:          %[[TILED_ATTENTION:.+]]:3 = iree_linalg_ext.attention 
// TILING-SAME:                                           ins(%[[Q_S]], %[[K_S]], %[[V_S]], %[[CST_1]]
// TILING-SAME:                                           outs(%[[ARG4]], %[[ARG5]], %[[ARG6]]
// TILING:          scf.yield %[[TILED_ATTENTION]]#0, %[[TILED_ATTENTION]]#1, %[[TILED_ATTENTION]]#2
// TILING:        }
// TILING:        %[[D7:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP]]], iterator_types = ["parallel",
// TILING-SAME:     "parallel"]} ins(%[[D6]]#2 : tensor<1024xf32>) outs(%[[D6]]#0 : tensor<1024x64xf32>)
// TILING-SAME:     {
// TILING:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILING-DAG:      %[[CST_1:.+]] = arith.constant 1.000000e+00 : f32
// TILING:          %[[D8:.+]] = arith.divf %[[CST_1]], %[[IN]] : f32
// TILING:          %[[D9:.+]] = arith.mulf %[[D8]], %[[OUT]] : f32
// TILING:          linalg.yield %[[D9]] : f32
// TILING:        } -> tensor<1024x64xf32>
// TILING:        %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D7]] into %[[D0]]
// TILING:        return %[[INSERTED_SLICE]] : tensor<1x1024x64xf32>
// TILING:      }

// -----

func.func @attention(%query: tensor<?x?x?xf32>, %key: tensor<?x?x?xf32>, %value: tensor<?x?x?xf32>, %dim0: index, %dim1: index, %dim2: index) -> tensor<?x?x?xf32> {
  %0 = tensor.empty(%dim0, %dim1, %dim2) : tensor<?x?x?xf32>
  %scale = arith.constant 0.05 : f32
  %1 = iree_linalg_ext.attention ins(%query, %key, %value, %scale : tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, f32) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}

// TILING-DAG:  #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// TILING-DAG:  #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
// TILING-LABEL:      func.func @attention(
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
// TILING:          %[[TILED_ATTENTION:.+]]:3 = iree_linalg_ext.attention ins(%[[EXTRACTED_SLICE_4]], %[[EXTRACTED_SLICE]], %[[EXTRACTED_SLICE_3]], %{{[a-z0-1]+}} :
// TILING-SAME:                      outs(%[[ARG7]], %[[ARG8]], %[[ARG9]] :
// TILING-SAME:                      -> tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>
// TILING:          scf.yield %[[TILED_ATTENTION]]#0, %[[TILED_ATTENTION]]#1, %[[TILED_ATTENTION]]#2 : tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>
// TILING:        }
// TILING:        %[[D7:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP]]], iterator_types = ["parallel",
// TILING-SAME:     "parallel"]} ins(%[[D6]]#[[D2:.+]] : tensor<?xf32>) outs(%[[D6]]#[[D0:.+]] : tensor<?x?xf32>) {
// TILING:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILING-DAG:      %[[CST_3:.+]] = arith.constant 1.000000e+00 : f32
// TILING:          %[[D8:.+]] = arith.divf %[[CST_3]], %[[IN]] : f32
// TILING:          %[[D9:.+]] = arith.mulf %[[D8]], %[[OUT]] : f32
// TILING:          linalg.yield %[[D9]] : f32
// TILING:        } -> tensor<?x?xf32>
// TILING:        %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D7]] into %[[D0]][0, 0, 0] [1, %[[DIM]], %[[DIM_0]]]
// TILING-SAME:     [1, 1, 1] : tensor<?x?xf32> into tensor<?x?x?xf32>
// TILING:        return %[[INSERTED_SLICE]] : tensor<?x?x?xf32>
// TILING:      }

// -----

func.func @attention(%query: tensor<1x1024x64xf16>, %key: tensor<1x1024x64xf16>, %value: tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16> {
  %0 = tensor.empty() : tensor<1x1024x64xf16>
  %scale = arith.constant 0.05 : f16
  %1 = iree_linalg_ext.attention ins(%query, %key, %value, %scale : tensor<1x1024x64xf16>, tensor<1x1024x64xf16>, tensor<1x1024x64xf16>, f16) outs(%0 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
  return %1 : tensor<1x1024x64xf16>
}

// TILING-DAG:  #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// TILING-LABEL:      @attention(
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
// TILING:          %[[TILED_ATTENTION:.+]]:3 = iree_linalg_ext.attention ins(%[[EXTRACTED_SLICE_3]], %[[EXTRACTED_SLICE_1]], %[[EXTRACTED_SLICE_2]], %{{[a-z0-9]+}} :
// TILING-SAME:                                           outs(%[[ARG4]], %[[ARG5]], %[[ARG6]] :
// TILING-SAME:                                           -> tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>
// TILING:          scf.yield %[[TILED_ATTENTION]]#0, %[[TILED_ATTENTION]]#1, %[[TILED_ATTENTION]]#2 : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>
// TILING:        }
// TILING:        %[[D7:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP]]], iterator_types = ["parallel",
// TILING-SAME:     "parallel"]} ins(%[[D6]]#[[D2:.+]] : tensor<1024xf32>) outs(%[[D6]]#[[D0:.+]] : tensor<1024x64xf32>)
// TILING-SAME:     {
// TILING:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILING-DAG:      %[[CST_1:.+]] = arith.constant 1.000000e+00 : f32
// TILING:          %[[D9:.+]] = arith.divf %[[CST_1]], %[[IN]] : f32
// TILING:          %[[D10:.+]] = arith.mulf %[[D9]], %[[OUT]] : f32
// TILING:          linalg.yield %[[D10]] : f32
// TILING:        } -> tensor<1024x64xf32>
// TILING:        %[[D8:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel",
// TILING-SAME:     "parallel"]} ins(%[[D7]] : tensor<1024x64xf32>) outs(%[[EXTRACTED_SLICE]] : tensor<1024x64xf16>) {
// TILING:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f16):
// TILING:          %[[D9]] = arith.truncf %[[IN]] : f32 to f16
// TILING:          linalg.yield %[[D9]] : f16
// TILING:        } -> tensor<1024x64xf16>
// TILING:        %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D8]] into %[[D0]][0, 0, 0] [1, 1024, 64] [1, 1, 1] :
// TILING-SAME:     tensor<1024x64xf16> into tensor<1x1024x64xf16>
// TILING:        return %[[INSERTED_SLICE]] : tensor<1x1024x64xf16>
// TILING:      }

// -----

func.func @attention_transpose_v(%query: tensor<1x1024x64xf16>, %key: tensor<1x1024x64xf16>, %value: tensor<1x64x1024xf16>) -> tensor<1x1024x64xf16> {
  %0 = tensor.empty() : tensor<1x1024x64xf16>
  %scale = arith.constant 0.05 : f16
  %1 = iree_linalg_ext.attention {transpose_v = true} ins(%query, %key, %value, %scale : tensor<1x1024x64xf16>, tensor<1x1024x64xf16>, tensor<1x64x1024xf16>, f16) outs(%0 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
  return %1 : tensor<1x1024x64xf16>
}
// TILING-DAG:  #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0)>
// TILING-DAG:  #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// TILING-LABEL:      func.func @attention_transpose_v
// TILING-SAME:        (%[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x1024x64xf16>, %[[ARG1:[a-zA-Z0-9_]+]]:
// TILING-SAME:   tensor<1x1024x64xf16>, %[[ARG2:[a-zA-Z0-9_]+]]: tensor<1x64x1024xf16>) -> tensor<1x1024x64xf16> {
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
// TILING:          %[[EXTRACTED_SLICE_1:.+]] = tensor.extract_slice %[[ARG1]][0, %[[ARG3]], 0] [1, 1024, 64] [1, 1, 1] :
// TILING-SAME:       tensor<1x1024x64xf16> to tensor<1024x64xf16>
// TILING:          %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[ARG2]][0, 0, %[[ARG3]]] [1, 64, 1024] [1, 1, 1] :
// TILING-SAME:       tensor<1x64x1024xf16> to tensor<64x1024xf16>
// TILING:          %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[ARG0]][0, 0, 0] [1, 1024, 64] [1, 1, 1] :
// TILING-SAME:       tensor<1x1024x64xf16> to tensor<1024x64xf16>
// TILING:          %[[D9:.+]]:3 = iree_linalg_ext.attention {transpose_v = true} ins(%[[EXTRACTED_SLICE_3]],
// TILING-SAME:       %[[EXTRACTED_SLICE_1]], %[[EXTRACTED_SLICE_2]], %{{.+}} : tensor<1024x64xf16>, tensor<1024x64xf16>,
// TILING-SAME:       tensor<64x1024xf16>, f16) outs(%[[ARG4]], %[[ARG5]], %[[ARG6]] : tensor<1024x64xf32>, tensor<1024xf32>,
// TILING-SAME:       tensor<1024xf32>) -> tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>
// TILING:          scf.yield %[[D9]]#[[D0:.+]], %[[D9]]#[[D1:.+]], %[[D9]]#[[D2:.+]] : tensor<1024x64xf32>,
// TILING-SAME:       tensor<1024xf32>, tensor<1024xf32>
// TILING:        }
// TILING:        %[[D7:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]]], iterator_types = ["parallel",
// TILING-SAME:     "parallel"]} ins(%[[D6]]#[[D2]] : tensor<1024xf32>) outs(%[[D6]]#[[D0]] : tensor<1024x64xf32>) {
// TILING:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILING-DAG:      %[[CST_1:.+]] = arith.constant 1.000000e+00 : f32
// TILING:          %[[D9]] = arith.divf %[[CST_1]], %[[IN]] : f32
// TILING:          %[[D10:.+]] = arith.mulf %[[D9]], %[[OUT]] : f32
// TILING:          linalg.yield %[[D10]] : f32
// TILING:        } -> tensor<1024x64xf32>
// TILING:        %[[D8:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel",
// TILING-SAME:     "parallel"]} ins(%[[D7]] : tensor<1024x64xf32>) outs(%[[EXTRACTED_SLICE]] : tensor<1024x64xf16>) {
// TILING:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f16):
// TILING:          %[[D9]] = arith.truncf %[[IN]] : f32 to f16
// TILING:          linalg.yield %[[D9]] : f16
// TILING:        } -> tensor<1024x64xf16>
// TILING:        %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D8]] into %[[D0]][0, 0, 0] [1, 1024, 64] [1, 1, 1] :
// TILING-SAME:     tensor<1024x64xf16> into tensor<1x1024x64xf16>
// TILING:        return %[[INSERTED_SLICE]] : tensor<1x1024x64xf16>
