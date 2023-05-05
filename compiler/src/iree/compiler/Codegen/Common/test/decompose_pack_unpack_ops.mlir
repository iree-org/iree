// RUN: iree-opt --iree-codegen-decompose-pack-unpack-ops --split-input-file %s | FileCheck %s

func.func @simple_KCRS_to_KCRSsr(%arg0: tensor<1x1x32x8xf32>, %arg1: tensor<1x1x1x1x8x32xf32>) -> tensor<1x1x1x1x8x32xf32> {
  %0 = tensor.pack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : tensor<1x1x32x8xf32> -> tensor<1x1x1x1x8x32xf32>
  return %0 : tensor<1x1x1x1x8x32xf32>
}
// CHECK-LABEL: func.func @simple_KCRS_to_KCRSsr
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK:         %[[TILE:.+]] = tensor.extract_slice %[[IN]][0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1] : tensor<1x1x32x8xf32> to tensor<32x8xf32>
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<8x32xf32>
// CHECK:         %[[TRANS:.+]] = linalg.transpose ins(%[[TILE]] : tensor<32x8xf32>) outs(%[[EMPTY]] : tensor<8x32xf32>) permutation = [1, 0]
// CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[TRANS]] into %[[OUT]][0, 0, 0, 0, 0, 0] [1, 1, 1, 1, 8, 32] [1, 1, 1, 1, 1, 1]
// CHECK:         return %[[INSERT]]

// -----

func.func @simple_pad_and_pack(%input: tensor<5x1xf32>, %output: tensor<1x1x8x2xf32>, %pad: f32) -> tensor<1x1x8x2xf32> {
  %0 = tensor.pack %input padding_value(%pad : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %output : tensor<5x1xf32> -> tensor<1x1x8x2xf32>
  return %0 : tensor<1x1x8x2xf32>
}
// CHECK-LABEL: func.func @simple_pad_and_pack
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[PAD_VAL:[A-Za-z0-9]+]]:
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK:         %[[PAD:.+]] = tensor.pad %[[IN]] low[%[[C0]], %[[C0]]] high[%[[C3]], %[[C1]]]
// CHECK:           tensor.yield %[[PAD_VAL]]
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<8x2xf32>
// CHECK:         %[[TRANS:.+]] = linalg.transpose ins(%[[PAD]] : tensor<8x2xf32>) outs(%[[EMPTY:.+]] : tensor<8x2xf32>) permutation = [0, 1]
// CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[TRANS]] into %[[OUT]][0, 0, 0, 0] [1, 1, 8, 2] [1, 1, 1, 1]
// CHECK:         return %[[INSERT]]

// -----

func.func @simple_NC_to_CNnc(%arg0: tensor<32x8xf32>, %arg1: tensor<1x1x32x8xf32>) -> tensor<1x1x32x8xf32>{
  %0 = tensor.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : tensor<32x8xf32> -> tensor<1x1x32x8xf32>
  return %0 : tensor<1x1x32x8xf32>
}
// CHECK-LABEL: func.func @simple_NC_to_CNnc
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<32x8xf32>
// CHECK:         %[[TRANS:.+]] = linalg.transpose ins(%[[IN]] : tensor<32x8xf32>) outs(%[[EMPTY:.+]] : tensor<32x8xf32>) permutation = [0, 1]
// CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[TRANS]] into %[[OUT]][0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK:         return %[[INSERT]]

// -----

func.func @KCRS_to_KCRSsr(%arg0: tensor<1x1x128x64xf32>, %arg1: tensor<1x1x4x8x8x32xf32>) -> tensor<1x1x4x8x8x32xf32> {
  %0 = tensor.pack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : tensor<1x1x128x64xf32> -> tensor<1x1x4x8x8x32xf32>
  return %0 : tensor<1x1x4x8x8x32xf32>
}
// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0) -> (d0 * 8)>
// CHECK:       func.func @KCRS_to_KCRSsr
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:     %[[C8:.+]] = arith.constant 8 : index
// CHECK:         %[[RES0:.+]] = scf.for %[[I:.+]] = %[[C0]] to %[[C4]] step %[[C1]]
// CHECK-SAME:      iter_args(%[[ITER0:.+]] = %[[OUT]])
// CHECK:           %[[RES1:.+]] = scf.for %[[J:.+]] = %[[C0]] to %[[C8]] step %[[C1]]
// CHECK-SAME:        iter_args(%[[ITER1:.+]] = %[[ITER0]])
// CHECK-DAG:         %[[IDX2:.+]] = affine.apply #[[MAP0]](%[[I]])
// CHECK-DAG:         %[[IDX3:.+]] = affine.apply #[[MAP1]](%[[J]])
// CHECK:             %[[IN_SLICE:.+]] = tensor.extract_slice %[[IN]]
// CHECK-SAME:          [0, 0, %[[IDX2]], %[[IDX3]]] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK:             %[[ITER_SLICE:.+]] = tensor.extract_slice %[[ITER1]]
// CHECK-SAME:          [0, 0, %[[I]], %[[J]], 0, 0] [1, 1, 1, 1, 8, 32] [1, 1, 1, 1, 1, 1]
// CHECK-SAME:          tensor<1x1x4x8x8x32xf32> to tensor<1x1x1x1x8x32xf32>
// CHECK:             %[[TILE:.+]] = tensor.extract_slice %[[IN_SLICE]]
// CHECK-SAME:          [0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK-SAME:          tensor<1x1x32x8xf32> to tensor<32x8xf32>
// CHECK:             %[[EMPTY:.+]] = tensor.empty() : tensor<8x32xf32>
// CHECK:             %[[TRANS:.+]] = linalg.transpose ins(%[[TILE]] : tensor<32x8xf32>) outs(%[[EMPTY]] : tensor<8x32xf32>) permutation = [1, 0]
// CHECK:             %[[INSERT0:.+]] = tensor.insert_slice %[[TRANS]] into %[[ITER_SLICE]][0, 0, 0, 0, 0, 0] [1, 1, 1, 1, 8, 32] [1, 1, 1, 1, 1, 1]
// CHECK:             %[[INSERT1:.+]] = tensor.insert_slice %[[INSERT0]] into %[[ITER1]]
// CHECK-SAME:          [0, 0, %[[I]], %[[J]], 0, 0]
// CHECK:             scf.yield %[[INSERT1]]
// CHECK:           }
// CHECK:           scf.yield %[[RES1]]
// CHECK:         }
// CHECK:         return %[[RES0]]

// -----

func.func @pad_and_pack(%arg0: tensor<13x15xf32>, %arg1: tensor<2x8x8x2xf32>, %arg2: f32) -> tensor<2x8x8x2xf32> {
  %0 = tensor.pack %arg0 padding_value(%arg2 : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %arg1 : tensor<13x15xf32> -> tensor<2x8x8x2xf32>
  return %0 : tensor<2x8x8x2xf32>
}
// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0) -> (d0 * 8)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0) -> (d0 * -8 + 13, 8)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-DAG:   #[[MAP3:.+]] = affine_map<(d0) -> (d0 * -2 + 15, 2)>
// CHECK-DAG:   #[[MAP4:.+]] = affine_map<(d0) -> (-d0 + 8)>
// CHECK-DAG:   #[[MAP5:.+]] = affine_map<(d0) -> (-d0 + 2)>
// CHECK:       func.func @pad_and_pack
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[PAD_VAL:[A-Za-z0-9]+]]:
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C8:.+]] = arith.constant 8 : index
// CHECK:         %[[RES0:.+]] = scf.for %[[I:.+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-SAME:      iter_args(%[[ITER0:.+]] = %[[OUT]])
// CHECK:           %[[RES1:.+]] = scf.for %[[J:.+]] = %[[C0]] to %[[C8]] step %[[C1]]
// CHECK-SAME:        iter_args(%[[ITER1:.+]] = %[[ITER0]])
// CHECK-DAG:         %[[IDX0:.+]] = affine.apply #[[MAP0]](%[[I]])
// CHECK-DAG:         %[[SZ0:.+]] = affine.min #[[MAP1]](%[[I]])
// CHECK-DAG:         %[[IDX1:.+]] = affine.apply #[[MAP2]](%[[J]])
// CHECK-DAG:         %[[SZ1:.+]] = affine.min #[[MAP3]](%[[J]])
// CHECK:             %[[TILE:.+]] = tensor.extract_slice %[[IN]][%[[IDX0]], %[[IDX1]]] [%[[SZ0]], %[[SZ1]]]
// CHECK:             %[[ITER_SLICE:.+]] = tensor.extract_slice %[[ITER1]]
// CHECK-SAME:          [%[[I]], %[[J]], 0, 0] [1, 1, 8, 2] [1, 1, 1, 1]
// CHECK:             %[[H0:.+]] = affine.apply #[[MAP4]](%[[SZ0]])
// CHECK:             %[[H1:.+]] = affine.apply #[[MAP5]](%[[SZ1]])
// CHECK:             %[[PAD:.+]] = tensor.pad %[[TILE]] low[%[[C0]], %[[C0]]] high[%[[H0]], %[[H1]]]
// CHECK:               tensor.yield %[[PAD_VAL]]
// CHECK:             %[[EMPTY:.+]] = tensor.empty() : tensor<8x2xf32>
// CHECK:             %[[TRANS:.+]] = linalg.transpose ins(%[[PAD]] : tensor<8x2xf32>) outs(%[[EMPTY:.+]] : tensor<8x2xf32>) permutation = [0, 1]
// CHECK:             %[[INSERT0:.+]] = tensor.insert_slice %[[TRANS]] into %[[ITER_SLICE]][0, 0, 0, 0] [1, 1, 8, 2] [1, 1, 1, 1]
// CHECK:             %[[INSERT1:.+]] = tensor.insert_slice %[[INSERT0]] into %[[ITER1]]
// CHECK-SAME:          [%[[I]], %[[J]], 0, 0]
// CHECK:             scf.yield %[[INSERT1]]
// CHECK:           }
// CHECK:           scf.yield %[[RES1]]
// CHECK:         }
// CHECK:         return %[[RES0]]

// -----

func.func @KC_to_CKck(%arg0: tensor<128x256xf32>, %arg1: tensor<32x4x32x8xf32>) -> tensor<32x4x32x8xf32> {
  %0 = tensor.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : tensor<128x256xf32> -> tensor<32x4x32x8xf32>
  return %0 : tensor<32x4x32x8xf32>
}
// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0) -> (d0 * 8)>
// CHECK:       func.func @KC_to_CKck
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : index
// CHECK:         %[[RES0:.+]] = scf.for %[[C:.+]] = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK-SAME:      iter_args(%[[ITER0:.+]] = %[[OUT]])
// CHECK:           %[[RES1:.+]] = scf.for %[[K:.+]] = %[[C0]] to %[[C4]] step %[[C1]]
// CHECK-SAME:        iter_args(%[[ITER1:.+]] = %[[ITER0]])
// CHECK-DAG:         %[[IN_K:.+]] = affine.apply #[[MAP0]](%[[K]])
// CHECK-DAG:         %[[IN_C:.+]] = affine.apply #[[MAP1]](%[[C]])
// CHECK:             %[[TILE:.+]] = tensor.extract_slice %[[IN]][%[[IN_K]], %[[IN_C]]] [32, 8] [1, 1]
// CHECK:             %[[ITER_SLICE:.+]] = tensor.extract_slice %[[ITER1]]
// CHECK-SAME:          [%[[C]], %[[K]], 0, 0] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK:             %[[EMPTY:.+]] = tensor.empty() : tensor<32x8xf32>
// CHECK:             %[[TRANS:.+]] = linalg.transpose ins(%[[TILE]] : tensor<32x8xf32>) outs(%[[EMPTY:.+]] : tensor<32x8xf32>) permutation = [0, 1]
// CHECK:             %[[INSERT:.+]] = tensor.insert_slice %[[TRANS]] into %[[ITER_SLICE]][0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK:             %[[INSERT1:.+]] = tensor.insert_slice %[[INSERT0]] into %[[ITER1]]
// CHECK-SAME:          [%[[C]], %[[K]], 0, 0] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK:             scf.yield %[[INSERT1]]
// CHECK:           }
// CHECK:           scf.yield %[[RES1]]
// CHECK:         }
// CHECK:         return %[[RES0]]

// -----

func.func @simple_KCRSsr_to_KCRS(%arg0: tensor<1x1x1x1x8x32xf32>, %arg1: tensor<1x1x32x8xf32>) -> tensor<1x1x32x8xf32> {
  %0 = tensor.unpack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : tensor<1x1x1x1x8x32xf32> -> tensor<1x1x32x8xf32>
  return %0 : tensor<1x1x32x8xf32>
}
// CHECK-LABEL: func.func @simple_KCRSsr_to_KCRS
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK:         %[[TILE:.+]] = tensor.extract_slice %[[IN]]
// CHECK-SAME:      [0, 0, 0, 0, 0, 0] [1, 1, 1, 1, 8, 32] [1, 1, 1, 1, 1, 1]
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<32x8xf32>
// CHECK:         %[[TRANS:.+]] = linalg.transpose ins(%[[TILE]] : tensor<8x32xf32>) outs(%[[EMPTY]] : tensor<32x8xf32>) permutation = [1, 0]
// CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[TRANS]] into %[[OUT]][0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK:         return %[[INSERT]]

// -----

func.func @simple_unpack_and_extract_slice(%input: tensor<1x1x8x2xf32>, %output: tensor<5x1xf32>) -> tensor<5x1xf32> {
  %0 = tensor.unpack %input inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %output : tensor<1x1x8x2xf32> -> tensor<5x1xf32>
  return %0 : tensor<5x1xf32>
}
// CHECK-LABEL: func.func @simple_unpack_and_extract_slice
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK:         %[[TILE:.+]] = tensor.extract_slice %[[IN]][0, 0, 0, 0] [1, 1, 8, 2] [1, 1, 1, 1]
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<8x2xf32>
// CHECK:         %[[TRANS:.+]] = linalg.transpose ins(%[[TILE]] : tensor<8x2xf32>) outs(%[[EMPTY]] : tensor<8x2xf32>) permutation = [0, 1]
// CHECK:         %[[RES:.+]] = tensor.extract_slice %[[TRANS]][0, 0] [5, 1] [1, 1]
// CHECK:         return %[[RES:.+]]

// -----

func.func @simple_CNnc_to_NC(%arg0: tensor<1x1x32x8xf32>, %arg1: tensor<32x8xf32>) -> tensor<32x8xf32>{
  %0 = tensor.unpack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : tensor<1x1x32x8xf32> -> tensor<32x8xf32>
  return %0 : tensor<32x8xf32>
}
// CHECK-LABEL: func.func @simple_CNnc_to_NC
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK:         %[[TILE:.+]] = tensor.extract_slice %[[IN]][0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<32x8xf32>
// CHECK:         %[[TRANS:.+]] = linalg.transpose ins(%[[TILE]] : tensor<32x8xf32>) outs(%[[EMPTY]] : tensor<32x8xf32>) permutation = [0, 1]
// CHECK:         return %[[TRANS]]

// -----

func.func @KCRSsr_to_KCRS(%arg0: tensor<13x12x4x8x8x32xf32>, %arg1: tensor<13x12x128x64xf32>) -> tensor<13x12x128x64xf32> {
  %0 = tensor.unpack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : tensor<13x12x4x8x8x32xf32> -> tensor<13x12x128x64xf32>
  return %0 : tensor<13x12x128x64xf32>
}
// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0) -> (d0 floordiv 32)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0) -> (d0 floordiv 8)>
// CHECK:       func.func @KCRSsr_to_KCRS
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:     %[[C12:.+]] = arith.constant 12 : index
// CHECK-DAG:     %[[C13:.+]] = arith.constant 13 : index
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:     %[[C128:.+]] = arith.constant 128 : index
// CHECK:         %[[RES0:.+]] = scf.for %[[K:.+]] = %[[C0]] to %[[C13]] step %[[C1]]
// CHECK-SAME:      iter_args(%[[ITER0:.+]] = %[[OUT]])
// CHECK:         %[[RES1:.+]] = scf.for %[[C:.+]] = %[[C0]] to %[[C12]] step %[[C1]]
// CHECK-SAME:      iter_args(%[[ITER1:.+]] = %[[ITER0]])
// CHECK:         %[[RES2:.+]] = scf.for %[[R:.+]] = %[[C0]] to %[[C128]] step %[[C32]]
// CHECK-SAME:      iter_args(%[[ITER2:.+]] = %[[ITER1]])
// CHECK:           %[[RES3:.+]] = scf.for %[[S:.+]] = %[[C0]] to %[[C64]] step %[[C8]]
// CHECK-SAME:        iter_args(%[[ITER3:.+]] = %[[ITER2]])
// CHECK-DAG:         %[[IN_R:.+]] = affine.apply #[[MAP0]](%[[R]])
// CHECK-DAG:         %[[IN_S:.+]] = affine.apply #[[MAP1]](%[[S]])
// CHECK-DAG:         %[[IN_SLICE:.+]] = tensor.extract_slice %[[IN]]
// CHECK-SAME:          [%[[K]], %[[C]], %[[IN_R]], %[[IN_S]], 0, 0] [1, 1, 1, 1, 8, 32] [1, 1, 1, 1, 1, 1]
// CHECK-DAG:         %[[ITER3_SLICE:.+]] = tensor.extract_slice %[[ITER3]]
// CHECK-SAME:          [%[[K]], %[[C]], %[[R]], %[[S]]] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK-DAG:         %[[TILE:.+]] = tensor.extract_slice %[[IN_SLICE]]
// CHECK-SAME:          [0, 0, 0, 0, 0, 0] [1, 1, 1, 1, 8, 32] [1, 1, 1, 1, 1, 1]
// CHECK:             %[[EMPTY:.+]] = tensor.empty() : tensor<32x8xf32>
// CHECK:             %[[TRANS:.+]] = linalg.transpose ins(%[[TILE]] : tensor<8x32xf32>) outs(%[[EMPTY]] : tensor<32x8xf32>) permutation = [1, 0]
// CHECK:             %[[INSERT0:.+]] = tensor.insert_slice %[[TRANS]] into %[[ITER3_SLICE]][0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK:             %[[INSERT1:.+]] = tensor.insert_slice %[[INSERT0]]
// CHECK-SAME:          into %[[ITER3]][%[[K]], %[[C]], %[[R]], %[[S]]] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK:             scf.yield %[[INSERT1]]
// CHECK:           }
// CHECK:           scf.yield %[[RES3]]
// CHECK:         }
// CHECK:         scf.yield %[[RES2]]
// CHECK:         scf.yield %[[RES1]]
// CHECK:         return %[[RES0]]

// -----

func.func @unpack_and_extract_slice(%arg0: tensor<2x8x8x2xf32>, %arg1: tensor<13x15xf32>) -> tensor<13x15xf32> {
  %0 = tensor.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %arg1 : tensor<2x8x8x2xf32> -> tensor<13x15xf32>
  return %0 : tensor<13x15xf32>
}
// CHECK-DAG:  #[[MAP0:.+]] = affine_map<(d0) -> (-d0 + 13, 8)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0) -> (-d0 + 15, 2)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0 floordiv 8)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0) -> (d0 floordiv 2)>
// CHECK:      func.func @unpack_and_extract_slice
// CHECK-SAME:   %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:   %[[OUT:[A-Za-z0-9]+]]:
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:    %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:    %[[C13:.+]] = arith.constant 13 : index
// CHECK-DAG:    %[[C15:.+]] = arith.constant 15 : index
// CHECK:        %[[RES0:.+]] = scf.for %[[I:.+]] = %[[C0]] to %[[C13]] step %[[C8]]
// CHECK-SAME:     iter_args(%[[ITER0:.+]] = %[[OUT]])
// CHECK-DAG:      %[[OUT_I_SZ:.+]] = affine.min #[[MAP0]](%[[I]])
// CHECK:          %[[RES1:.+]] = scf.for %[[J:.+]] = %[[C0]] to %[[C15]] step %[[C2]]
// CHECK-SAME:       iter_args(%[[ITER1:.+]] = %[[ITER0]])
// CHECK-DAG:        %[[OUT_J_SZ:.+]] = affine.min #[[MAP1]](%[[J]])
// CHECK-DAG:        %[[IN_I:.+]] = affine.apply #[[MAP2]](%[[I]])
// CHECK-DAG:        %[[IN_J:.+]] = affine.apply #[[MAP3]](%[[J]])
// CHECK-DAG:        %[[IN_SLICE]] = tensor.extract_slice %[[IN]]
// CHECK-SAME:         [%[[IN_I]], %[[IN_J]], 0, 0] [1, 1, 8, 2] [1, 1, 1, 1]
// CHECK-DAG:        %[[ITER1_SLICE1:.+]] = tensor.extract_slice %[[ITER1]]
// CHECK-SAME:         [%[[I]], %[[J]]] [%[[OUT_I_SZ]], %[[OUT_J_SZ]]] [1, 1]
// CHECK:            %[[TILE:.+]] = tensor.extract_slice %[[IN_SLICE]][0, 0, 0, 0] [1, 1, 8, 2] [1, 1, 1, 1]
// CHECK:            %[[EMPTY:.+]] = tensor.empty() : tensor<8x2xf32>
// CHECK:            %[[TRANS:.+]] = linalg.transpose ins(%[[TILE]] : tensor<8x2xf32>) outs(%[[EMPTY]] : tensor<8x2xf32>) permutation = [0, 1]
// CHECK:            %[[TRANS_SLICE:.+]] = tensor.extract_slice %[[TRANS]][0, 0] [%[[OUT_I_SZ]], %[[OUT_J_SZ]]] [1, 1]
// CHECK:            %[[INSERT1:.+]] = tensor.insert_slice %[[TRANS_SLICE]]
// CHECK-SAME:         into %[[ITER1_SLICE1]][0, 0] [%[[OUT_I_SZ]], %[[OUT_J_SZ]]] [1, 1]
// CHECK:            %[[INSERT2:.+]] = tensor.insert_slice %[[INSERT1]]
// CHECK-SAME:         into %[[ITER1]][%[[I]], %[[J]]] [%[[OUT_I_SZ]], %[[OUT_J_SZ]]] [1, 1]
// CHECK:            scf.yield %[[INSERT2]]
// CHECK:          }
// CHECK:          scf.yield %[[RES1]]
// CHECK:        }
// CHECK:        return %[[RES0]]

// -----

func.func @CKck_to_KC(%arg0: tensor<32x4x32x8xf32>, %arg1: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %0 = tensor.unpack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : tensor<32x4x32x8xf32> -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}
// CHECK-DAG:  #[[MAP0:.+]] = affine_map<(d0) -> (d0 floordiv 32)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0) -> (d0 floordiv 8)>
// CHECK:      func.func @CKck_to_KC
// CHECK-SAME:   %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:   %[[OUT:[A-Za-z0-9]+]]:
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:    %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:    %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:    %[[C256:.+]] = arith.constant 256 : index
// CHECK:        %[[RES0:.+]] = scf.for %[[K:.+]] = %[[C0]] to %[[C128]] step %[[C32]]
// CHECK-SAME:     iter_args(%[[ITER0:.+]] = %[[OUT]])
// CHECK:          %[[RES1:.+]] = scf.for %[[C:.+]] = %[[C0]] to %[[C256]] step %[[C8]]
// CHECK-SAME:       iter_args(%[[ITER1:.+]] = %[[ITER0]])
// CHECK-DAG:        %[[IN_K:.+]] = affine.apply #[[MAP0]](%[[K]])
// CHECK-DAG:        %[[IN_C:.+]] = affine.apply #[[MAP1]](%[[C]])
// CHECK:            %[[IN_SLICE:.+]] = tensor.extract_slice %[[IN]][%[[IN_C]], %[[IN_K]], 0, 0] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK:            %[[TILE:.+]] = tensor.extract_slice %[[IN_SLICE]][0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1] : tensor<1x1x32x8xf32> to tensor<32x8xf32>
// CHECK:            %[[EMPTY:.+]] = tensor.empty() : tensor<32x8xf32>
// CHECK:            %[[TRANS:.+]] = linalg.transpose ins(%[[TILE]] : tensor<32x8xf32>) outs(%[[EMPTY]] : tensor<32x8xf32>) permutation = [0, 1]
// CHECK:            %[[INSERT:.+]] = tensor.insert_slice %[[TRANS]] into %[[ITER1]][%[[K]], %[[C]]] [32, 8] [1, 1]
// CHECK:            scf.yield %[[INSERT]]
// CHECK:          }
// CHECK:          scf.yield %[[RES1]]
// CHECK:        }
// CHECK:        return %[[RES0]]
