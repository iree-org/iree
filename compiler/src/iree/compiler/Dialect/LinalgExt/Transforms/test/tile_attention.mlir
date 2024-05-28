// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-linalg-ext-tile-attention),cse)" %s | FileCheck %s --check-prefix=CHECK

// TODO: These tests should be moved to tiling.mlir when PartialReductionOpInterface is implemented for attention op.

func.func @attention(%query: tensor<1x1024x64xf32>, %key: tensor<1x1024x64xf32>, %value: tensor<1x1024x64xf32>) -> tensor<1x1024x64xf32> {
  %0 = tensor.empty() : tensor<1x1024x64xf32>
  %scale = arith.constant 0.05 : f32
  %1 = iree_linalg_ext.attention ins(%query, %key, %value, %scale : tensor<1x1024x64xf32>, tensor<1x1024x64xf32>, tensor<1x1024x64xf32>, f32) outs(%0 : tensor<1x1024x64xf32>) -> tensor<1x1024x64xf32>
  return %1 : tensor<1x1024x64xf32>
}

// CHECK-DAG:  #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:  #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL:      func.func @attention
// CHECK-SAME: (%[[QUERY:.+]]: tensor<1x1024x64xf32>, %[[KEY:.+]]: tensor<1x1024x64xf32>, %[[VALUE:.+]]: tensor<1x1024x64xf32>)
// CHECK-DAG:    %[[D0:.+]] = tensor.empty() : tensor<1x1024x64xf32>
// CHECK-DAG:    %[[D1:.+]] = tensor.empty() : tensor<1024x64xf32>
// CHECK-DAG:    %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:    %[[CST_1:.+]] = arith.constant 5.000000e-02 : f32
// CHECK:        %[[D2:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D1]] : tensor<1024x64xf32>)
// CHECK-DAG:    %[[CST_0:.+]] = arith.constant -1.000000e+30 : f32
// CHECK:        %[[D3:.+]] = tensor.empty() : tensor<1024xf32>
// CHECK:        %[[D4:.+]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[D3]] : tensor<1024xf32>)
// CHECK:        %[[D5:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D3]] : tensor<1024xf32>)
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
// CHECK:        %[[D6:.+]]:3 = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C1024]] step %[[C1024]]
// CHECK-SAME:     iter_args(%[[ARG4:.+]] = %[[D2]], %[[ARG5:.+]] = %[[D4]], %[[ARG6:.+]] = %[[D5]])
// CHECK:          %[[K_S:.+]] = tensor.extract_slice %[[KEY]][0, %[[ARG3]], 0] [1, 1024, 64] [1, 1, 1]
// CHECK:          %[[V_S:.+]] = tensor.extract_slice %[[VALUE]][0, %[[ARG3]], 0] [1, 1024, 64] [1, 1, 1]
// CHECK:          %[[Q_S:.+]] = tensor.extract_slice %[[QUERY]][0, 0, 0] [1, 1024, 64] [1, 1, 1]
// CHECK:          %[[ATT:.+]]:3 = iree_linalg_ext.attention 
// CHECK-SAME:                                           ins(%[[Q_S]], %[[K_S]], %[[V_S]], %[[CST_1]]
// CHECK-SAME:                                           outs(%[[ARG4]], %[[ARG5]], %[[ARG6]]
// CHECK:          scf.yield %[[ATT]]#0, %[[ATT]]#1, %[[ATT]]#2
// CHECK:        }
// CHECK:        %[[D7:.+]] = linalg.generic 
// CHECK-SAME:                {indexing_maps = [#[[$MAP1]], #[[$MAP]]], 
// CHECK-SAME:                 iterator_types = ["parallel", "parallel"]} 
// CHECK-SAME:               ins(%[[D6]]#2 : tensor<1024xf32>) 
// CHECK-SAME:               outs(%[[D6]]#0 : tensor<1024x64xf32>) {
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK-DAG:      %[[CST_1:.+]] = arith.constant 1.000000e+00
// CHECK:          %[[D8:.+]] = arith.divf %[[CST_1]], %[[IN]]
// CHECK:          %[[D9:.+]] = arith.mulf %[[D8]], %[[OUT]]
// CHECK:          linalg.yield %[[D9]]
// CHECK:        } -> tensor<1024x64xf32>
// CHECK:        %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D7]] into %[[D0]]
// CHECK:        return %[[INSERTED_SLICE]] : tensor<1x1024x64xf32>
// CHECK:      }

// -----

func.func @attention(%query: tensor<?x?x?xf32>, %key: tensor<?x?x?xf32>, %value: tensor<?x?x?xf32>, %dim0: index, %dim1: index, %dim2: index) -> tensor<?x?x?xf32> {
  %0 = tensor.empty(%dim0, %dim1, %dim2) : tensor<?x?x?xf32>
  %scale = arith.constant 0.05 : f32
  %1 = iree_linalg_ext.attention ins(%query, %key, %value, %scale : tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, f32) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}

// CHECK-DAG:  #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:  #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL:      func.func @attention(
// CHECK-SAME:  %[[QUERY:.+]]: tensor<?x?x?xf32>, %[[KEY:.+]]: tensor<?x?x?xf32>, %[[VALUE:.+]]: tensor<?x?x?xf32>,
// CHECK-SAME:  %[[ARG3:[a-zA-Z0-9_]+]]: index, %[[ARG4:[a-zA-Z0-9_]+]]: index, %[[ARG5:[a-zA-Z0-9_]+]]: index)
// CHECK:        %[[D0:.+]] = tensor.empty(%[[ARG3]], %[[ARG4]], %[[ARG5]]) : tensor<?x?x?xf32>
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK:        %[[DIM:.+]] = tensor.dim %[[QUERY]], %[[C1]] : tensor<?x?x?xf32>
// CHECK-DAG:    %[[C2:.+]] = arith.constant 2 : index
// CHECK:        %[[DIM_0:.+]] = tensor.dim %[[QUERY]], %[[C2]] : tensor<?x?x?xf32>
// CHECK:        %[[DIM_1:.+]] = tensor.dim %[[KEY]], %[[C1]] : tensor<?x?x?xf32>
// CHECK:        %[[D1:.+]] = tensor.empty(%[[DIM]], %[[DIM_0]]) : tensor<?x?xf32>
// CHECK-DAG:    %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[D2:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D1]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-DAG:    %[[CST_2:.+]] = arith.constant -1.000000e+30 : f32
// CHECK:        %[[D3:.+]] = tensor.empty(%[[DIM]]) : tensor<?xf32>
// CHECK:        %[[D4:.+]] = linalg.fill ins(%[[CST_2]] : f32) outs(%[[D3]] : tensor<?xf32>) -> tensor<?xf32>
// CHECK:        %[[D5:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D3]] : tensor<?xf32>) -> tensor<?xf32>
// CHECK:        %[[D6:.+]]:3 = scf.for %[[ARG6:[a-zA-Z0-9_]+]] = %[[C0]] to %[[DIM_1]] step %[[DIM]]
// CHECK-SAME:     iter_args(%[[ARG7:[a-zA-Z0-9_]+]] = %[[D2]], %[[ARG8:[a-zA-Z0-9_]+]] = %[[D4]],
// CHECK-SAME:     %[[ARG9:[a-zA-Z0-9_]+]] = %[[D5]]) -> (tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>) {
// CHECK:          %[[K_S:.+]] = tensor.extract_slice %[[KEY]][0, %[[ARG6]], 0] [1, %[[DIM]], %[[DIM_0]]] [1, 1, 1]
// CHECK:          %[[V_S:.+]] = tensor.extract_slice %[[VALUE]][0, %[[ARG6]], 0] [1, %[[DIM]], %[[DIM_0]]] [1, 1, 1]
// CHECK:          %[[Q_S:.+]] = tensor.extract_slice %[[QUERY]][0, 0, 0] [1, %[[DIM]], %[[DIM_0]]] [1, 1, 1]
// CHECK:          %[[ATT:.+]]:3 = iree_linalg_ext.attention ins(%[[Q_S]], %[[K_S]], %[[V_S]], %{{[a-z0-1]+}}
// CHECK-SAME:                      outs(%[[ARG7]], %[[ARG8]], %[[ARG9]]
// CHECK:          scf.yield %[[ATT]]#0, %[[ATT]]#1, %[[ATT]]#2
// CHECK:        }
// CHECK:        %[[D7:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP]]], iterator_types = ["parallel",
// CHECK-SAME:     "parallel"]} ins(%[[D6]]#[[D2:.+]] : tensor<?xf32>) outs(%[[D6]]#[[D0:.+]] : tensor<?x?xf32>) {
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK-DAG:      %[[CST_3:.+]] = arith.constant 1.000000e+00
// CHECK:          %[[D8:.+]] = arith.divf %[[CST_3]], %[[IN]]
// CHECK:          %[[D9:.+]] = arith.mulf %[[D8]], %[[OUT]]
// CHECK:          linalg.yield %[[D9]]
// CHECK:        } -> tensor<?x?xf32>
// CHECK:        %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D7]] into %[[D0]][0, 0, 0] [1, %[[DIM]], %[[DIM_0]]]
// CHECK-SAME:     [1, 1, 1] : tensor<?x?xf32> into tensor<?x?x?xf32>
// CHECK:        return %[[INSERTED_SLICE]] : tensor<?x?x?xf32>
// CHECK:      }

// -----

func.func @attention_f16(%query: tensor<1x1024x64xf16>, %key: tensor<1x1024x64xf16>, %value: tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16> {
  %0 = tensor.empty() : tensor<1x1024x64xf16>
  %scale = arith.constant 0.05 : f16
  %1 = iree_linalg_ext.attention ins(%query, %key, %value, %scale : tensor<1x1024x64xf16>, tensor<1x1024x64xf16>, tensor<1x1024x64xf16>, f16) outs(%0 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
  return %1 : tensor<1x1024x64xf16>
}

// CHECK-LABEL:  @attention_f16

// CHECK:        scf.for
// CHECK:          iree_linalg_ext.attention ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<1024x64xf16>, tensor<1024x64xf16>, tensor<1024x64xf16>, f16
// CHECK-SAME:                                           outs(%{{.*}}, %{{.*}}, %{{.*}} :
// CHECK-SAME:                                           -> tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>
// CHECK:        scf.yield

// CHECK:        linalg.generic

// CHECK:        %[[TRUNCED:.+]] = linalg.generic
// CHECK:          arith.truncf
// CHECK:        } -> tensor<1024x64xf16>

// CHECK:        %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[TRUNCED]]
// CHECK:        return %[[INSERTED_SLICE]]
// CHECK:      }

// -----

func.func @attention_transpose_v(%query: tensor<1x1024x64xf16>, %key: tensor<1x1024x64xf16>, %value: tensor<1x64x1024xf16>) -> tensor<1x1024x64xf16> {
  %0 = tensor.empty() : tensor<1x1024x64xf16>
  %scale = arith.constant 0.05 : f16
  %1 = iree_linalg_ext.attention {transpose_v = true} ins(%query, %key, %value, %scale : tensor<1x1024x64xf16>, tensor<1x1024x64xf16>, tensor<1x64x1024xf16>, f16) outs(%0 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
  return %1 : tensor<1x1024x64xf16>
}
// CHECK-LABEL:  func.func @attention_transpose_v
// CHECK:          scf.for
// CHECK:            iree_linalg_ext.attention {transpose_v = true}
// CHECK:          scf.yield 
