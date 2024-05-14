// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-linalg-ext-tile-and-decompose-attention{tileSize=32}),cse)" %s | FileCheck %s --check-prefix=TILESIZE

func.func @attention(%query: tensor<1x1024x64xf32>, %key: tensor<1x1024x64xf32>, %value: tensor<1x1024x64xf32>) -> tensor<1x1024x64xf32> {
  %0 = tensor.empty() : tensor<1x1024x64xf32>
  %scale = arith.constant 0.05 : f32
  %1 = iree_linalg_ext.attention ins(%query, %key, %value, %scale : tensor<1x1024x64xf32>, tensor<1x1024x64xf32>, tensor<1x1024x64xf32>, f32) outs(%0 : tensor<1x1024x64xf32>) -> tensor<1x1024x64xf32>
  return %1 : tensor<1x1024x64xf32>
}

// TILESIZE-DAG:  #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// TILESIZE-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
// TILESIZE-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0)>
// TILESIZE:      func.func @attention
// TILESIZE-SAME: (%[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x1024x64xf32>, %[[ARG1:[a-zA-Z0-9_]+]]:
// TILESIZE-SAME:   tensor<1x1024x64xf32>, %[[ARG2:[a-zA-Z0-9_]+]]: tensor<1x1024x64xf32>) -> tensor<1x1024x64xf32> {
// TILESIZE:        %[[D0:.+]] = tensor.empty() : tensor<1x1024x64xf32>
// TILESIZE:        %[[D1:.+]] = tensor.empty() : tensor<1024x64xf32>
// TILESIZE-DAG:    %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// TILESIZE:        %[[D2:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D1]] : tensor<1024x64xf32>) ->
// TILESIZE-SAME:     tensor<1024x64xf32>
// TILESIZE-DAG:    %[[CST_0:.+]] = arith.constant -1.000000e+30 : f32
// TILESIZE:        %[[D3:.+]] = tensor.empty() : tensor<1024xf32>
// TILESIZE:        %[[D4:.+]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[D3]] : tensor<1024xf32>) -> tensor<1024xf32>
// TILESIZE:        %[[D5:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D3]] : tensor<1024xf32>) -> tensor<1024xf32>
// TILESIZE-DAG:    %[[C0:.+]] = arith.constant 0 : index
// TILESIZE-DAG:    %[[C32:.+]] = arith.constant 32 : index
// TILESIZE-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
// TILESIZE:        %[[D6:.+]]:3 = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1024]] step %[[C32]]
// TILESIZE-SAME:     iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[D2]], %[[ARG5:[a-zA-Z0-9_]+]] = %[[D4]],
// TILESIZE-SAME:     %[[ARG6:[a-zA-Z0-9_]+]] = %[[D5]]) -> (tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>) {
// TILESIZE:          %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG1]][0, %[[ARG3]], 0] [1, 32, 64] [1, 1, 1] :
// TILESIZE-SAME:       tensor<1x1024x64xf32> to tensor<32x64xf32>
// TILESIZE:          %[[EXTRACTED_SLICE_1:.+]] = tensor.extract_slice %[[ARG2]][0, %[[ARG3]], 0] [1, 32, 64] [1, 1, 1] :
// TILESIZE-SAME:       tensor<1x1024x64xf32> to tensor<32x64xf32>
// TILESIZE:          %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[ARG0]][0, 0, 0] [1, 1024, 64] [1, 1, 1] :
// TILESIZE-SAME:       tensor<1x1024x64xf32> to tensor<1024x64xf32>
// TILESIZE:          %[[SCALE_Q:.+]] = linalg.generic {{.+}} ins(%[[EXTRACTED_SLICE_2]] : tensor<1024x64xf32>)
// TILESIZE:          %[[D8:.+]] = tensor.empty() : tensor<1024x32xf32>
// TILESIZE:          %[[D9:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D8]] : tensor<1024x32xf32>) ->
// TILESIZE-SAME:       tensor<1024x32xf32>
// TILESIZE:          %[[D10:.+]] = linalg.matmul_transpose_b ins(%[[SCALE_Q]], %[[EXTRACTED_SLICE]] :
// TILESIZE-SAME:       tensor<1024x64xf32>, tensor<32x64xf32>) outs(%[[D9]] : tensor<1024x32xf32>) -> tensor<1024x32xf32>
// TILESIZE:          %[[D11:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]]], iterator_types = ["parallel",
// TILESIZE-SAME:       "reduction"]} ins(%[[D10]] : tensor<1024x32xf32>) outs(%[[ARG5]] : tensor<1024xf32>) {
// TILESIZE:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE:            %[[D18:.+]] = arith.maximumf %[[IN]], %[[OUT]] : f32
// TILESIZE:            linalg.yield %[[D18]] : f32
// TILESIZE:          } -> tensor<1024xf32>
// TILESIZE:          %[[D12:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP]]], iterator_types = ["parallel",
// TILESIZE-SAME:       "parallel"]} ins(%[[D11]] : tensor<1024xf32>) outs(%[[D10]] : tensor<1024x32xf32>) {
// TILESIZE:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE:            %[[D18]] = arith.subf %[[OUT]], %[[IN]] : f32
// TILESIZE:            %[[D19:.+]] = math.exp2 %[[D18]] : f32
// TILESIZE:            linalg.yield %[[D19]] : f32
// TILESIZE:          } -> tensor<1024x32xf32>
// TILESIZE:          %[[D13:.+]] = linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP2]]], iterator_types = ["parallel"]}
// TILESIZE-SAME:       ins(%[[D11]] : tensor<1024xf32>) outs(%[[ARG5]] : tensor<1024xf32>) {
// TILESIZE:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE:            %[[D18]] = arith.subf %[[OUT]], %[[IN]] : f32
// TILESIZE:            %[[D19]] = math.exp2 %[[D18]] : f32
// TILESIZE:            linalg.yield %[[D19]] : f32
// TILESIZE:          } -> tensor<1024xf32>
// TILESIZE:          %[[D14:.+]] = linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP2]]], iterator_types = ["parallel"]}
// TILESIZE-SAME:       ins(%[[D13]] : tensor<1024xf32>) outs(%[[ARG6]] : tensor<1024xf32>) {
// TILESIZE:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE:            %[[D18]] = arith.mulf %[[IN]], %[[OUT]] : f32
// TILESIZE:            linalg.yield %[[D18]] : f32
// TILESIZE:          } -> tensor<1024xf32>
// TILESIZE:          %[[D15:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]]], iterator_types = ["parallel",
// TILESIZE-SAME:       "reduction"]} ins(%[[D12]] : tensor<1024x32xf32>) outs(%[[D14]] : tensor<1024xf32>) {
// TILESIZE:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE:            %[[D18]] = arith.addf %[[IN]], %[[OUT]] : f32
// TILESIZE:            linalg.yield %[[D18]] : f32
// TILESIZE:          } -> tensor<1024xf32>
// TILESIZE:          %[[D16:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP]]], iterator_types = ["parallel",
// TILESIZE-SAME:       "parallel"]} ins(%[[D13]] : tensor<1024xf32>) outs(%[[ARG4]] : tensor<1024x64xf32>) {
// TILESIZE:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE:            %[[D18]] = arith.mulf %[[IN]], %[[OUT]] : f32
// TILESIZE:            linalg.yield %[[D18]] : f32
// TILESIZE:          } -> tensor<1024x64xf32>
// TILESIZE:          %[[D17:.+]] = linalg.matmul ins(%[[D12]], %[[EXTRACTED_SLICE_1]] : tensor<1024x32xf32>,
// TILESIZE-SAME:       tensor<32x64xf32>) outs(%[[D16]] : tensor<1024x64xf32>) -> tensor<1024x64xf32>
// TILESIZE:          scf.yield %[[D17]], %[[D11]], %[[D15]] : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>
// TILESIZE:        }
// TILESIZE:        %[[D7:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP]]], iterator_types = ["parallel",
// TILESIZE-SAME:     "parallel"]} ins(%[[D6]]#[[D2:.+]] : tensor<1024xf32>) outs(%[[D6]]#[[D0:.+]] : tensor<1024x64xf32>)
// TILESIZE-SAME:     {
// TILESIZE:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE-DAG:      %[[CST_1:.+]] = arith.constant 1.000000e+00 : f32
// TILESIZE:          %[[D8:.+]] = arith.divf %[[CST_1]], %[[IN]] : f32
// TILESIZE:          %[[D9:.+]] = arith.mulf %[[D8]], %[[OUT]] : f32
// TILESIZE:          linalg.yield %[[D9]] : f32
// TILESIZE:        } -> tensor<1024x64xf32>
// TILESIZE:        %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D7]] into %[[D0]][0, 0, 0] [1, 1024, 64] [1, 1, 1] :
// TILESIZE-SAME:     tensor<1024x64xf32> into tensor<1x1024x64xf32>
// TILESIZE:        return %[[INSERTED_SLICE]] : tensor<1x1024x64xf32>
// TILESIZE:      }

// -----

func.func @attention(%query: tensor<?x?x?xf32>, %key: tensor<?x?x?xf32>, %value: tensor<?x?x?xf32>, %dim0: index, %dim1: index, %dim2: index) -> tensor<?x?x?xf32> {
  %0 = tensor.empty(%dim0, %dim1, %dim2) : tensor<?x?x?xf32>
  %scale = arith.constant 0.05 : f32
  %1 = iree_linalg_ext.attention ins(%query, %key, %value, %scale : tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, f32) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}

// TILESIZE-DAG:  #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// TILESIZE-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
// TILESIZE-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0)>
// TILESIZE:      func.func @attention(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>, %[[ARG1:[a-zA-Z0-9_]+]]:
// TILESIZE-SAME:   tensor<?x?x?xf32>, %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>, %[[ARG3:[a-zA-Z0-9_]+]]: index,
// TILESIZE-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: index, %[[ARG5:[a-zA-Z0-9_]+]]: index) -> tensor<?x?x?xf32> {
// TILESIZE:        %[[D0:.+]] = tensor.empty(%[[ARG3]], %[[ARG4]], %[[ARG5]]) : tensor<?x?x?xf32>
// TILESIZE-DAG:    %[[C0:.+]] = arith.constant 0 : index
// TILESIZE-DAG:    %[[C1:.+]] = arith.constant 1 : index
// TILESIZE:        %[[DIM:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?xf32>
// TILESIZE-DAG:    %[[C2:.+]] = arith.constant 2 : index
// TILESIZE:        %[[DIM_0:.+]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?xf32>
// TILESIZE:        %[[DIM_1:.+]] = tensor.dim %[[ARG1]], %[[C1]] : tensor<?x?x?xf32>
// TILESIZE:        %[[D1:.+]] = tensor.empty(%[[DIM]], %[[DIM_0]]) : tensor<?x?xf32>
// TILESIZE-DAG:    %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// TILESIZE:        %[[D2:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D1]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// TILESIZE-DAG:    %[[CST_2:.+]] = arith.constant -1.000000e+30 : f32
// TILESIZE:        %[[D3:.+]] = tensor.empty(%[[DIM]]) : tensor<?xf32>
// TILESIZE:        %[[D4:.+]] = linalg.fill ins(%[[CST_2]] : f32) outs(%[[D3]] : tensor<?xf32>) -> tensor<?xf32>
// TILESIZE:        %[[D5:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D3]] : tensor<?xf32>) -> tensor<?xf32>
// TILESIZE-DAG:    %[[C32:.+]] = arith.constant 32 : index
// TILESIZE:        %[[D6:.+]]:3 = scf.for %[[ARG6:[a-zA-Z0-9_]+]] = %[[C0]] to %[[DIM_1]] step %[[C32]]
// TILESIZE-SAME:     iter_args(%[[ARG7:[a-zA-Z0-9_]+]] = %[[D2]], %[[ARG8:[a-zA-Z0-9_]+]] = %[[D4]],
// TILESIZE-SAME:     %[[ARG9:[a-zA-Z0-9_]+]] = %[[D5]]) -> (tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>) {
// TILESIZE:          %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG1]][0, %[[ARG6]], 0] [1, 32, %[[DIM_0]]] [1, 1,
// TILESIZE-SAME:       1] : tensor<?x?x?xf32> to tensor<32x?xf32>
// TILESIZE:          %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[ARG2]][0, %[[ARG6]], 0] [1, 32, %[[DIM_0]]] [1,
// TILESIZE-SAME:       1, 1] : tensor<?x?x?xf32> to tensor<32x?xf32>
// TILESIZE:          %[[EXTRACTED_SLICE_4:.+]] = tensor.extract_slice %[[ARG0]][0, 0, 0] [1, %[[DIM]], %[[DIM_0]]] [1, 1,
// TILESIZE-SAME:       1] : tensor<?x?x?xf32> to tensor<?x?xf32>
// TILESIZE:          %[[DIM_5:.+]] = tensor.dim %[[EXTRACTED_SLICE_4]], %[[C0]] : tensor<?x?xf32>
// TILESIZE:          %[[SCALE_Q:.+]] = linalg.generic
// TILESIZE:          %[[D8:.+]] = tensor.empty(%[[DIM_5]]) : tensor<?x32xf32>
// TILESIZE:          %[[D9:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D8]] : tensor<?x32xf32>) -> tensor<?x32xf32>
// TILESIZE:          %[[D10:.+]] = linalg.matmul_transpose_b ins(%[[SCALE_Q]], %[[EXTRACTED_SLICE]] :
// TILESIZE-SAME:       tensor<?x?xf32>, tensor<32x?xf32>) outs(%[[D9]] : tensor<?x32xf32>) -> tensor<?x32xf32>
// TILESIZE:          %[[D11:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]]], iterator_types = ["parallel",
// TILESIZE-SAME:       "reduction"]} ins(%[[D10]] : tensor<?x32xf32>) outs(%[[ARG8]] : tensor<?xf32>) {
// TILESIZE:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE:            %[[D18:.+]] = arith.maximumf %[[IN]], %[[OUT]] : f32
// TILESIZE:            linalg.yield %[[D18]] : f32
// TILESIZE:          } -> tensor<?xf32>
// TILESIZE:          %[[D12:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP]]], iterator_types = ["parallel",
// TILESIZE-SAME:       "parallel"]} ins(%[[D11]] : tensor<?xf32>) outs(%[[D10]] : tensor<?x32xf32>) {
// TILESIZE:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE:            %[[D18]] = arith.subf %[[OUT]], %[[IN]] : f32
// TILESIZE:            %[[D19:.+]] = math.exp2 %[[D18]] : f32
// TILESIZE:            linalg.yield %[[D19]] : f32
// TILESIZE:          } -> tensor<?x32xf32>
// TILESIZE:          %[[D13:.+]] = linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP2]]], iterator_types = ["parallel"]}
// TILESIZE-SAME:       ins(%[[D11]] : tensor<?xf32>) outs(%[[ARG8]] : tensor<?xf32>) {
// TILESIZE:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE:            %[[D18]] = arith.subf %[[OUT]], %[[IN]] : f32
// TILESIZE:            %[[D19]] = math.exp2 %[[D18]] : f32
// TILESIZE:            linalg.yield %[[D19]] : f32
// TILESIZE:          } -> tensor<?xf32>
// TILESIZE:          %[[D14:.+]] = linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP2]]], iterator_types = ["parallel"]}
// TILESIZE-SAME:       ins(%[[D13]] : tensor<?xf32>) outs(%[[ARG9]] : tensor<?xf32>) {
// TILESIZE:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE:            %[[D18]] = arith.mulf %[[IN]], %[[OUT]] : f32
// TILESIZE:            linalg.yield %[[D18]] : f32
// TILESIZE:          } -> tensor<?xf32>
// TILESIZE:          %[[D15:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]]], iterator_types = ["parallel",
// TILESIZE-SAME:       "reduction"]} ins(%[[D12]] : tensor<?x32xf32>) outs(%[[D14]] : tensor<?xf32>) {
// TILESIZE:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE:            %[[D18]] = arith.addf %[[IN]], %[[OUT]] : f32
// TILESIZE:            linalg.yield %[[D18]] : f32
// TILESIZE:          } -> tensor<?xf32>
// TILESIZE:          %[[D16:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP]]], iterator_types = ["parallel",
// TILESIZE-SAME:       "parallel"]} ins(%[[D13]] : tensor<?xf32>) outs(%[[ARG7]] : tensor<?x?xf32>) {
// TILESIZE:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE:            %[[D18]] = arith.mulf %[[IN]], %[[OUT]] : f32
// TILESIZE:            linalg.yield %[[D18]] : f32
// TILESIZE:          } -> tensor<?x?xf32>
// TILESIZE:          %[[D17:.+]] = linalg.matmul ins(%[[D12]], %[[EXTRACTED_SLICE_3]] : tensor<?x32xf32>,
// TILESIZE-SAME:       tensor<32x?xf32>) outs(%[[D16]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// TILESIZE:          scf.yield %[[D17]], %[[D11]], %[[D15]] : tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>
// TILESIZE:        }
// TILESIZE:        %[[D7:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP]]], iterator_types = ["parallel",
// TILESIZE-SAME:     "parallel"]} ins(%[[D6]]#[[D2:.+]] : tensor<?xf32>) outs(%[[D6]]#[[D0:.+]] : tensor<?x?xf32>) {
// TILESIZE:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE-DAG:      %[[CST_3:.+]] = arith.constant 1.000000e+00 : f32
// TILESIZE:          %[[D8:.+]] = arith.divf %[[CST_3]], %[[IN]] : f32
// TILESIZE:          %[[D9:.+]] = arith.mulf %[[D8]], %[[OUT]] : f32
// TILESIZE:          linalg.yield %[[D9]] : f32
// TILESIZE:        } -> tensor<?x?xf32>
// TILESIZE:        %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D7]] into %[[D0]][0, 0, 0] [1, %[[DIM]], %[[DIM_0]]]
// TILESIZE-SAME:     [1, 1, 1] : tensor<?x?xf32> into tensor<?x?x?xf32>
// TILESIZE:        return %[[INSERTED_SLICE]] : tensor<?x?x?xf32>
// TILESIZE:      }

// -----

func.func @attention(%query: tensor<1x1024x64xf16>, %key: tensor<1x1024x64xf16>, %value: tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16> {
  %0 = tensor.empty() : tensor<1x1024x64xf16>
  %scale = arith.constant 0.05 : f16
  %1 = iree_linalg_ext.attention ins(%query, %key, %value, %scale : tensor<1x1024x64xf16>, tensor<1x1024x64xf16>, tensor<1x1024x64xf16>, f16) outs(%0 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
  return %1 : tensor<1x1024x64xf16>
}

// TILESIZE-DAG:  #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// TILESIZE-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
// TILESIZE-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0)>
// TILESIZE:      func.func @attention(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x1024x64xf16>, %[[ARG1:[a-zA-Z0-9_]+]]:
// TILESIZE-SAME:   tensor<1x1024x64xf16>, %[[ARG2:[a-zA-Z0-9_]+]]: tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16> {
// TILESIZE:        %[[D0:.+]] = tensor.empty() : tensor<1x1024x64xf16>
// TILESIZE:        %[[D1:.+]] = tensor.empty() : tensor<1024x64xf32>
// TILESIZE:        %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[D0]][0, 0, 0] [1, 1024, 64] [1, 1, 1] :
// TILESIZE-SAME:     tensor<1x1024x64xf16> to tensor<1024x64xf16>
// TILESIZE-DAG:    %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// TILESIZE:        %[[D2:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D1]] : tensor<1024x64xf32>) ->
// TILESIZE-SAME:     tensor<1024x64xf32>
// TILESIZE-DAG:    %[[CST_0:.+]] = arith.constant -1.000000e+30 : f32
// TILESIZE:        %[[D3:.+]] = tensor.empty() : tensor<1024xf32>
// TILESIZE:        %[[D4:.+]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[D3]] : tensor<1024xf32>) -> tensor<1024xf32>
// TILESIZE:        %[[D5:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D3]] : tensor<1024xf32>) -> tensor<1024xf32>
// TILESIZE-DAG:    %[[C0:.+]] = arith.constant 0 : index
// TILESIZE-DAG:    %[[C32:.+]] = arith.constant 32 : index
// TILESIZE-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
// TILESIZE:        %[[D6:.+]]:3 = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1024]] step %[[C32]]
// TILESIZE-SAME:     iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[D2]], %[[ARG5:[a-zA-Z0-9_]+]] = %[[D4]],
// TILESIZE-SAME:     %[[ARG6:[a-zA-Z0-9_]+]] = %[[D5]]) -> (tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>) {
// TILESIZE:          %[[EXTRACTED_SLICE_1:.+]] = tensor.extract_slice %[[ARG1]][0, %[[ARG3]], 0] [1, 32, 64] [1, 1, 1] :
// TILESIZE-SAME:       tensor<1x1024x64xf16> to tensor<32x64xf16>
// TILESIZE:          %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[ARG2]][0, %[[ARG3]], 0] [1, 32, 64] [1, 1, 1] :
// TILESIZE-SAME:       tensor<1x1024x64xf16> to tensor<32x64xf16>
// TILESIZE:          %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[ARG0]][0, 0, 0] [1, 1024, 64] [1, 1, 1] :
// TILESIZE-SAME:       tensor<1x1024x64xf16> to tensor<1024x64xf16>
// TILESIZE:          %[[SCALE_Q:.+]] = linalg.generic
// TILESIZE:          %[[D9:.+]] = tensor.empty() : tensor<1024x32xf32>
// TILESIZE:          %[[D10:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D9]] : tensor<1024x32xf32>) ->
// TILESIZE-SAME:       tensor<1024x32xf32>
// TILESIZE:          %[[D11:.+]] = linalg.matmul_transpose_b ins(%[[SCALE_Q]], %[[EXTRACTED_SLICE_1]] :
// TILESIZE-SAME:       tensor<1024x64xf16>, tensor<32x64xf16>) outs(%[[D10]] : tensor<1024x32xf32>) ->
// TILESIZE-SAME:       tensor<1024x32xf32>
// TILESIZE:          %[[D12:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]]], iterator_types = ["parallel",
// TILESIZE-SAME:       "reduction"]} ins(%[[D11]] : tensor<1024x32xf32>) outs(%[[ARG5]] : tensor<1024xf32>) {
// TILESIZE:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE:            %[[D21:.+]] = arith.maximumf %[[IN]], %[[OUT]] : f32
// TILESIZE:            linalg.yield %[[D21]] : f32
// TILESIZE:          } -> tensor<1024xf32>
// TILESIZE:          %[[D13:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP]]], iterator_types = ["parallel",
// TILESIZE-SAME:       "parallel"]} ins(%[[D12]] : tensor<1024xf32>) outs(%[[D11]] : tensor<1024x32xf32>) {
// TILESIZE:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE:            %[[D21]] = arith.subf %[[OUT]], %[[IN]] : f32
// TILESIZE:            %[[D22:.+]] = math.exp2 %[[D21]] : f32
// TILESIZE:            linalg.yield %[[D22]] : f32
// TILESIZE:          } -> tensor<1024x32xf32>
// TILESIZE:          %[[D14:.+]] = linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP2]]], iterator_types = ["parallel"]}
// TILESIZE-SAME:       ins(%[[D12]] : tensor<1024xf32>) outs(%[[ARG5]] : tensor<1024xf32>) {
// TILESIZE:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE:            %[[D21]] = arith.subf %[[OUT]], %[[IN]] : f32
// TILESIZE:            %[[D22]] = math.exp2 %[[D21]] : f32
// TILESIZE:            linalg.yield %[[D22]] : f32
// TILESIZE:          } -> tensor<1024xf32>
// TILESIZE:          %[[D15:.+]] = linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP2]]], iterator_types = ["parallel"]}
// TILESIZE-SAME:       ins(%[[D14]] : tensor<1024xf32>) outs(%[[ARG6]] : tensor<1024xf32>) {
// TILESIZE:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE:            %[[D21]] = arith.mulf %[[IN]], %[[OUT]] : f32
// TILESIZE:            linalg.yield %[[D21]] : f32
// TILESIZE:          } -> tensor<1024xf32>
// TILESIZE:          %[[D16:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]]], iterator_types = ["parallel",
// TILESIZE-SAME:       "reduction"]} ins(%[[D13]] : tensor<1024x32xf32>) outs(%[[D15]] : tensor<1024xf32>) {
// TILESIZE:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE:            %[[D21]] = arith.addf %[[IN]], %[[OUT]] : f32
// TILESIZE:            linalg.yield %[[D21]] : f32
// TILESIZE:          } -> tensor<1024xf32>
// TILESIZE:          %[[D17:.+]] = tensor.empty() : tensor<1024x32xf16>
// TILESIZE:          %[[D18:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel",
// TILESIZE-SAME:       "parallel"]} ins(%[[D13]] : tensor<1024x32xf32>) outs(%[[D17]] : tensor<1024x32xf16>) {
// TILESIZE:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f16):
// TILESIZE:            %[[D21]] = arith.truncf %[[IN]] : f32 to f16
// TILESIZE:            linalg.yield %[[D21]] : f16
// TILESIZE:          } -> tensor<1024x32xf16>
// TILESIZE:          %[[D19:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP]]], iterator_types = ["parallel",
// TILESIZE-SAME:       "parallel"]} ins(%[[D14]] : tensor<1024xf32>) outs(%[[ARG4]] : tensor<1024x64xf32>) {
// TILESIZE:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE:            %[[D21]] = arith.mulf %[[IN]], %[[OUT]] : f32
// TILESIZE:            linalg.yield %[[D21]] : f32
// TILESIZE:          } -> tensor<1024x64xf32>
// TILESIZE:          %[[D20:.+]] = linalg.matmul ins(%[[D18]], %[[EXTRACTED_SLICE_2]] : tensor<1024x32xf16>,
// TILESIZE-SAME:       tensor<32x64xf16>) outs(%[[D19]] : tensor<1024x64xf32>) -> tensor<1024x64xf32>
// TILESIZE:          scf.yield %[[D20]], %[[D12]], %[[D16]] : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>
// TILESIZE:        }
// TILESIZE:        %[[D7:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP]]], iterator_types = ["parallel",
// TILESIZE-SAME:     "parallel"]} ins(%[[D6]]#[[D2:.+]] : tensor<1024xf32>) outs(%[[D6]]#[[D0:.+]] : tensor<1024x64xf32>)
// TILESIZE-SAME:     {
// TILESIZE:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE-DAG:      %[[CST_1:.+]] = arith.constant 1.000000e+00 : f32
// TILESIZE:          %[[D9:.+]] = arith.divf %[[CST_1]], %[[IN]] : f32
// TILESIZE:          %[[D10:.+]] = arith.mulf %[[D9]], %[[OUT]] : f32
// TILESIZE:          linalg.yield %[[D10]] : f32
// TILESIZE:        } -> tensor<1024x64xf32>
// TILESIZE:        %[[D8:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel",
// TILESIZE-SAME:     "parallel"]} ins(%[[D7]] : tensor<1024x64xf32>) outs(%[[EXTRACTED_SLICE]] : tensor<1024x64xf16>) {
// TILESIZE:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f16):
// TILESIZE:          %[[D9]] = arith.truncf %[[IN]] : f32 to f16
// TILESIZE:          linalg.yield %[[D9]] : f16
// TILESIZE:        } -> tensor<1024x64xf16>
// TILESIZE:        %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D8]] into %[[D0]][0, 0, 0] [1, 1024, 64] [1, 1, 1] :
// TILESIZE-SAME:     tensor<1024x64xf16> into tensor<1x1024x64xf16>
// TILESIZE:        return %[[INSERTED_SLICE]] : tensor<1x1024x64xf16>
// TILESIZE:      }

// -----

func.func @attention_transpose_v(%query: tensor<1x1024x64xf16>, %key: tensor<1x1024x64xf16>, %value: tensor<1x64x1024xf16>) -> tensor<1x1024x64xf16> {
  %0 = tensor.empty() : tensor<1x1024x64xf16>
  %scale = arith.constant 0.05 : f16
  %1 = iree_linalg_ext.attention {transpose_v = true} ins(%query, %key, %value, %scale : tensor<1x1024x64xf16>, tensor<1x1024x64xf16>, tensor<1x64x1024xf16>, f16) outs(%0 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
  return %1 : tensor<1x1024x64xf16>
}

// TILESIZE-DAG:  #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// TILESIZE-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
// TILESIZE-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0)>
// TILESIZE:      func.func @attention_transpose_v(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x1024x64xf16>, %[[ARG1:[a-zA-Z0-9_]+]]:
// TILESIZE-SAME:   tensor<1x1024x64xf16>, %[[ARG2:[a-zA-Z0-9_]+]]: tensor<1x64x1024xf16>) -> tensor<1x1024x64xf16> {
// TILESIZE:        %[[D0:.+]] = tensor.empty() : tensor<1x1024x64xf16>
// TILESIZE:        %[[D1:.+]] = tensor.empty() : tensor<1024x64xf32>
// TILESIZE:        %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[D0]][0, 0, 0] [1, 1024, 64] [1, 1, 1] :
// TILESIZE-SAME:     tensor<1x1024x64xf16> to tensor<1024x64xf16>
// TILESIZE-DAG:    %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// TILESIZE:        %[[D2:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D1]] : tensor<1024x64xf32>) ->
// TILESIZE-SAME:     tensor<1024x64xf32>
// TILESIZE-DAG:    %[[CST_0:.+]] = arith.constant -1.000000e+30 : f32
// TILESIZE:        %[[D3:.+]] = tensor.empty() : tensor<1024xf32>
// TILESIZE:        %[[D4:.+]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[D3]] : tensor<1024xf32>) -> tensor<1024xf32>
// TILESIZE:        %[[D5:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D3]] : tensor<1024xf32>) -> tensor<1024xf32>
// TILESIZE-DAG:    %[[C0:.+]] = arith.constant 0 : index
// TILESIZE-DAG:    %[[C32:.+]] = arith.constant 32 : index
// TILESIZE-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
// TILESIZE:        %[[D6:.+]]:3 = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1024]] step %[[C32]]
// TILESIZE-SAME:     iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[D2]], %[[ARG5:[a-zA-Z0-9_]+]] = %[[D4]],
// TILESIZE-SAME:     %[[ARG6:[a-zA-Z0-9_]+]] = %[[D5]]) -> (tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>) {
// TILESIZE:          %[[EXTRACTED_SLICE_1:.+]] = tensor.extract_slice %[[ARG1]][0, %[[ARG3]], 0] [1, 32, 64] [1, 1, 1] :
// TILESIZE-SAME:       tensor<1x1024x64xf16> to tensor<32x64xf16>
// TILESIZE:          %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[ARG2]][0, 0, %[[ARG3]]] [1, 64, 32] [1, 1, 1] :
// TILESIZE-SAME:       tensor<1x64x1024xf16> to tensor<64x32xf16>
// TILESIZE:          %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[ARG0]][0, 0, 0] [1, 1024, 64] [1, 1, 1] :
// TILESIZE-SAME:       tensor<1x1024x64xf16> to tensor<1024x64xf16>
// TILESIZE:          %[[SCALE_Q:.+]] = linalg.generic
// TILESIZE:          %[[D9:.+]] = tensor.empty() : tensor<1024x32xf32>
// TILESIZE:          %[[D10:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D9]] : tensor<1024x32xf32>) ->
// TILESIZE-SAME:       tensor<1024x32xf32>
// TILESIZE:          %[[D11:.+]] = linalg.matmul_transpose_b ins(%[[SCALE_Q]], %[[EXTRACTED_SLICE_1]] :
// TILESIZE-SAME:       tensor<1024x64xf16>, tensor<32x64xf16>) outs(%[[D10]] : tensor<1024x32xf32>) ->
// TILESIZE-SAME:       tensor<1024x32xf32>
// TILESIZE:          %[[D12:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]]], iterator_types = ["parallel",
// TILESIZE-SAME:       "reduction"]} ins(%[[D11]] : tensor<1024x32xf32>) outs(%[[ARG5]] : tensor<1024xf32>) {
// TILESIZE:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE:            %[[D21:.+]] = arith.maximumf %[[IN]], %[[OUT]] : f32
// TILESIZE:            linalg.yield %[[D21]] : f32
// TILESIZE:          } -> tensor<1024xf32>
// TILESIZE:          %[[D13:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP]]], iterator_types = ["parallel",
// TILESIZE-SAME:       "parallel"]} ins(%[[D12]] : tensor<1024xf32>) outs(%[[D11]] : tensor<1024x32xf32>) {
// TILESIZE:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE:            %[[D21]] = arith.subf %[[OUT]], %[[IN]] : f32
// TILESIZE:            %[[D22:.+]] = math.exp2 %[[D21]] : f32
// TILESIZE:            linalg.yield %[[D22]] : f32
// TILESIZE:          } -> tensor<1024x32xf32>
// TILESIZE:          %[[D14:.+]] = linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP2]]], iterator_types = ["parallel"]}
// TILESIZE-SAME:       ins(%[[D12]] : tensor<1024xf32>) outs(%[[ARG5]] : tensor<1024xf32>) {
// TILESIZE:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE:            %[[D21]] = arith.subf %[[OUT]], %[[IN]] : f32
// TILESIZE:            %[[D22]] = math.exp2 %[[D21]] : f32
// TILESIZE:            linalg.yield %[[D22]] : f32
// TILESIZE:          } -> tensor<1024xf32>
// TILESIZE:          %[[D15:.+]] = linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP2]]], iterator_types = ["parallel"]}
// TILESIZE-SAME:       ins(%[[D14]] : tensor<1024xf32>) outs(%[[ARG6]] : tensor<1024xf32>) {
// TILESIZE:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE:            %[[D21]] = arith.mulf %[[IN]], %[[OUT]] : f32
// TILESIZE:            linalg.yield %[[D21]] : f32
// TILESIZE:          } -> tensor<1024xf32>
// TILESIZE:          %[[D16:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]]], iterator_types = ["parallel",
// TILESIZE-SAME:       "reduction"]} ins(%[[D13]] : tensor<1024x32xf32>) outs(%[[D15]] : tensor<1024xf32>) {
// TILESIZE:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE:            %[[D21]] = arith.addf %[[IN]], %[[OUT]] : f32
// TILESIZE:            linalg.yield %[[D21]] : f32
// TILESIZE:          } -> tensor<1024xf32>
// TILESIZE:          %[[D17:.+]] = tensor.empty() : tensor<1024x32xf16>
// TILESIZE:          %[[D18:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel",
// TILESIZE-SAME:       "parallel"]} ins(%[[D13]] : tensor<1024x32xf32>) outs(%[[D17]] : tensor<1024x32xf16>) {
// TILESIZE:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f16):
// TILESIZE:            %[[D21]] = arith.truncf %[[IN]] : f32 to f16
// TILESIZE:            linalg.yield %[[D21]] : f16
// TILESIZE:          } -> tensor<1024x32xf16>
// TILESIZE:          %[[D19:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP]]], iterator_types = ["parallel",
// TILESIZE-SAME:       "parallel"]} ins(%[[D14]] : tensor<1024xf32>) outs(%[[ARG4]] : tensor<1024x64xf32>) {
// TILESIZE:          ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE:            %[[D21]] = arith.mulf %[[IN]], %[[OUT]] : f32
// TILESIZE:            linalg.yield %[[D21]] : f32
// TILESIZE:          } -> tensor<1024x64xf32>
// TILESIZE:          %[[D20:.+]] = linalg.matmul_transpose_b ins(%[[D18]], %[[EXTRACTED_SLICE_2]] : tensor<1024x32xf16>,
// TILESIZE-SAME:       tensor<64x32xf16>) outs(%[[D19]] : tensor<1024x64xf32>) -> tensor<1024x64xf32>
// TILESIZE:          scf.yield %[[D20]], %[[D12]], %[[D16]] : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>
// TILESIZE:        }
// TILESIZE:        %[[D7:.+]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP]]], iterator_types = ["parallel",
// TILESIZE-SAME:     "parallel"]} ins(%[[D6]]#[[D2:.+]] : tensor<1024xf32>) outs(%[[D6]]#[[D0:.+]] : tensor<1024x64xf32>)
// TILESIZE-SAME:     {
// TILESIZE:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// TILESIZE-DAG:      %[[CST_1:.+]] = arith.constant 1.000000e+00 : f32
// TILESIZE:          %[[D9:.+]] = arith.divf %[[CST_1]], %[[IN]] : f32
// TILESIZE:          %[[D10:.+]] = arith.mulf %[[D9]], %[[OUT]] : f32
// TILESIZE:          linalg.yield %[[D10]] : f32
// TILESIZE:        } -> tensor<1024x64xf32>
// TILESIZE:        %[[D8:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel",
// TILESIZE-SAME:     "parallel"]} ins(%[[D7]] : tensor<1024x64xf32>) outs(%[[EXTRACTED_SLICE]] : tensor<1024x64xf16>) {
// TILESIZE:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f16):
// TILESIZE:          %[[D9]] = arith.truncf %[[IN]] : f32 to f16
// TILESIZE:          linalg.yield %[[D9]] : f16
// TILESIZE:        } -> tensor<1024x64xf16>
// TILESIZE:        %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D8]] into %[[D0]][0, 0, 0] [1, 1024, 64] [1, 1, 1] :
// TILESIZE-SAME:     tensor<1024x64xf16> into tensor<1x1024x64xf16>
// TILESIZE:        return %[[INSERTED_SLICE]] : tensor<1x1024x64xf16>
