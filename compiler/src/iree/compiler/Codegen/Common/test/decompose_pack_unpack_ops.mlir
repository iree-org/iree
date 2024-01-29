// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-decompose-pack-unpack-ops))" --split-input-file %s | FileCheck %s

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
// CHECK:         %[[PAD:.+]] = tensor.pad %[[IN]] low[0, 0] high[3, 1]
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
// CHECK:       func.func @KCRS_to_KCRSsr
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK:         %[[EXPAND:.+]] = tensor.expand_shape %[[IN]] {{\[}}[0], [1], [2, 3], [4, 5]] : tensor<1x1x128x64xf32> into tensor<1x1x4x32x8x8xf32>
// CHECK:          %[[TRANSP:.+]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPAND]] : tensor<1x1x4x32x8x8xf32>)
// CHECK-SAME:       outs(%[[OUT]] : tensor<1x1x4x8x8x32xf32>)
// CHECK-SAME:       permutation = [0, 1, 2, 4, 5, 3]
// CHECK:         return %[[TRANSP]]

// -----

func.func @pad_and_pack(%arg0: tensor<13x15xf32>, %arg1: tensor<2x8x8x2xf32>, %arg2: f32) -> tensor<2x8x8x2xf32> {
  %0 = tensor.pack %arg0 padding_value(%arg2 : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %arg1 : tensor<13x15xf32> -> tensor<2x8x8x2xf32>
  return %0 : tensor<2x8x8x2xf32>
}
// CHECK:       func.func @pad_and_pack
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[PAD_VAL:[A-Za-z0-9]+]]:
// CHECK:         %[[PAD:.+]] = tensor.pad %[[IN]] low[0, 0] high[3, 1]
// CHECK:           tensor.yield %[[PAD_VAL]]
// CHECK:         } : tensor<13x15xf32> to tensor<16x16xf32>
// CHECK:         %[[EXPAND:.+]] = tensor.expand_shape %[[PAD]] {{\[}}[0, 1], [2, 3]] : tensor<16x16xf32> into tensor<2x8x8x2xf32>
// CHECK:         %[[TRANS:.+]] = linalg.transpose
// CHECK-SAME:      ins(%[[EXPAND]] : tensor<2x8x8x2xf32>)
// CHECK-SAME:      outs(%[[OUT:.+]] : tensor<2x8x8x2xf32>)
// CHECK-SAME:      permutation = [0, 2, 1, 3]
// CHECK:         return %[[TRANSP]]

// -----

func.func @KC_to_CKck(%arg0: tensor<128x256xf32>, %arg1: tensor<32x4x32x8xf32>) -> tensor<32x4x32x8xf32> {
  %0 = tensor.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : tensor<128x256xf32> -> tensor<32x4x32x8xf32>
  return %0 : tensor<32x4x32x8xf32>
}
// CHECK:       func.func @KC_to_CKck
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK:         %[[EXPAND:.+]] = tensor.expand_shape %[[IN]] {{\[}}[0, 1], [2, 3]] : tensor<128x256xf32> into tensor<4x32x32x8xf32>
// CHECK:         %[[TRANSP:.+]] = linalg.transpose
// CHECK-SAME:      ins(%[[EXPAND]] : tensor<4x32x32x8xf32>)
// CHECK-SAME:      outs(%[[OUT]] : tensor<32x4x32x8xf32>)
// CHECK-SAME:      permutation = [2, 0, 1, 3]
// CHECK:         return %[[TRANSP]]

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
// CHECK:       func.func @KCRSsr_to_KCRS
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<13x12x4x32x8x8xf32>
// CHECK:         %[[TRANSP:.+]] = linalg.transpose
// CHECK-SAME:      ins(%[[IN]] : tensor<13x12x4x8x8x32xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<13x12x4x32x8x8xf32>)
// CHECK-SAME:      permutation = [0, 1, 2, 5, 3, 4]
// CHECK:         %[[COLLAPSE:.+]] = tensor.collapse_shape %[[TRANSP]]
// CHECK-SAME:      {{\[}}[0], [1], [2, 3], [4, 5]] : tensor<13x12x4x32x8x8xf32> into tensor<13x12x128x64xf32>
// CHECK:         %[[COPY:.]] = linalg.copy ins(%[[COLLAPSE]]
// CHECK-SAME:        outs(%[[OUT]]
// CHECK:         return %[[COPY]]

// -----

func.func @unpack_and_extract_slice(%arg0: tensor<2x8x8x2xf32>, %arg1: tensor<13x15xf32>) -> tensor<13x15xf32> {
  %0 = tensor.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %arg1 : tensor<2x8x8x2xf32> -> tensor<13x15xf32>
  return %0 : tensor<13x15xf32>
}
// CHECK:      func.func @unpack_and_extract_slice
// CHECK-SAME:   %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:   %[[OUT:[A-Za-z0-9]+]]:
// CHECK:          %[[EMPTY:.+]] = tensor.empty() : tensor<2x8x8x2xf32>
// CHECK:          %[[TRANSP:.+]] = linalg.transpose
// CHECK-SAME:       ins(%[[IN]] : tensor<2x8x8x2xf32>)
// CHECK-SAME:       outs(%[[EMPTY]] : tensor<2x8x8x2xf32>)
// CHECK-SAME:       permutation = [0, 2, 1, 3]
// CHECK:          %[[COLLAPSE:.+]] = tensor.collapse_shape %[[TRANSP]]
// CHECK-SAME:       {{\[}}[0, 1], [2, 3]] : tensor<2x8x8x2xf32> into tensor<16x16xf32>
// CHECK:          %[[SLICE:.+]] = tensor.extract_slice %[[COLLAPSE]]
// CHECK-SAME:       [0, 0] [13, 15] [1, 1] : tensor<16x16xf32> to tensor<13x15xf32>
// CHECK:          %[[COPY:.]] = linalg.copy ins(%[[SLICE]]
// CHECK-SAME:         outs(%[[OUT]]
// CHECK:          return %[[COPY]]
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

// -----

func.func @pack_matmul_DYN_LHS(%src: tensor<?x?xf32>, %dest: tensor<?x?x16x1xf32>) -> tensor<?x?x16x1xf32> {
  %pack = tensor.pack %src inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %dest : tensor<?x?xf32> -> tensor<?x?x16x1xf32>
  return %pack : tensor<?x?x16x1xf32>
}
// CHECK-DAG:  #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s0 * 16 - s1)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<()[s0, s1] -> (s0 - s1)>
// CHECK:      func.func @pack_matmul_DYN_LHS
// CHECK-SAME:   %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:   %[[OUT:[A-Za-z0-9]+]]:
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[D0:.+]] = tensor.dim %[[IN]], %c0 : tensor<?x?xf32>
// CHECK-DAG:    %[[H0:.+]] = affine.apply #[[MAP0]]
// CHECK-DAG:    %[[H1:.+]] = affine.apply #[[MAP1]]
// CHECK:        %[[PAD:.+]] = tensor.pad %[[IN]] low[0, 0] high[%[[H0]], %[[H1]]]
// CHECK:        %[[EXPANDED:.+]] = tensor.expand_shape %[[PAD]]
// CHECK-SAME:     {{\[}}[0, 1], [2, 3]] : tensor<?x?xf32> into tensor<?x16x?x1xf32>
// CHECK:        %[[TILE:.+]] = tensor.extract_slice %expanded
// CHECK-SAME:     : tensor<?x16x?x1xf32> to tensor<?x16x?xf32>
// CHECK:        %[[EMPTY:.+]] = tensor.empty({{.+}}) : tensor<?x?x16xf32>
// CHECK:        %[[TRANSP:.+]] = linalg.transpose
// CHECK-SAME:     ins(%[[TILE]] : tensor<?x16x?xf32>)
// CHECK-SAME:     outs(%[[EMPTY]] : tensor<?x?x16xf32>)
// CHECK-SAME:   permutation = [0, 2, 1]
// CHECK:        %{{.+}} = tensor.insert_slice %[[TRANSP]] into %[[OUT]]

// -----

func.func @pack_matmul_DYN_RHS(%src: tensor<?x?xf32>, %dest: tensor<?x?x16x1xf32>) -> tensor<?x?x16x1xf32> {
  %pack = tensor.pack %src outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 1] into %dest : tensor<?x?xf32> -> tensor<?x?x16x1xf32>
  return %pack : tensor<?x?x16x1xf32>
}
// CHECK-DAG:  #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s0 * 16 - s1)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<()[s0, s1] -> (s0 - s1)>
// CHECK:      func.func @pack_matmul_DYN_RHS
// CHECK-SAME:   %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:   %[[OUT:[A-Za-z0-9]+]]:
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[H0:.+]] = affine.apply #[[MAP1]]
// CHECK-DAG:    %[[H1:.+]] = affine.apply #[[MAP0]]
// CHECK:        %[[PAD:.+]] = tensor.pad %[[IN]] low[0, 0] high[%[[H0]], %[[H1]]]
// CHECK:        %[[EXPANDED:.+]] = tensor.expand_shape %[[PAD]]
// CHECK-SAME:     {{\[}}[0, 1], [2, 3]] : tensor<?x?xf32> into tensor<?x1x?x16xf32>
// CHECK:        %[[TRANSP:.+]] = linalg.transpose
// CHECK-SAME:     ins(%[[EXPANDED]] : tensor<?x1x?x16xf32>)
// CHECK-SAME:     outs(%[[OUT]] : tensor<?x?x16x1xf32>)
// CHECK-SAME:     permutation = [2, 0, 3, 1]
