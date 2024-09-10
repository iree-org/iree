// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-decompose-pack-unpack-ops))" --split-input-file %s | FileCheck %s -check-prefixes=CHECK-ALL,CHECK
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-decompose-pack-unpack-ops{use-only-reshapes=true}))" --split-input-file %s | FileCheck %s -check-prefixes=CHECK-ALL,CHECK-RESHAPE

func.func @simple_KCRS_to_KCRSsr(%arg0: tensor<1x1x32x8xf32>, %arg1: tensor<1x1x1x1x8x32xf32>) -> tensor<1x1x1x1x8x32xf32> {
  %0 = tensor.pack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : tensor<1x1x32x8xf32> -> tensor<1x1x1x1x8x32xf32>
  return %0 : tensor<1x1x1x1x8x32xf32>
}
// CHECK-ALL-LABEL: func.func @simple_KCRS_to_KCRSsr
// CHECK-ALL-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-ALL-SAME:    %[[OUT:[A-Za-z0-9]+]]:

// CHECK-RESHAPE:     %[[EXPANDED:.+]] = tensor.expand_shape %[[IN]] {{\[}}[0], [1], [2, 3], [4, 5]] output_shape [1, 1, 1, 32, 1, 8] : tensor<1x1x32x8xf32> into tensor<1x1x1x32x1x8xf32>
// CHECK-RESHAPE:     %[[RESULT:.+]] = linalg.transpose ins(%[[EXPANDED]] : tensor<1x1x1x32x1x8xf32>) outs(%[[OUT]] : tensor<1x1x1x1x8x32xf32>) permutation = [0, 1, 2, 4, 5, 3]

// CHECK:             %[[TILE:.+]] = tensor.extract_slice %[[IN]][0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1] : tensor<1x1x32x8xf32> to tensor<32x8xf32>
// CHECK:             %[[EMPTY:.+]] = tensor.empty() : tensor<8x32xf32>
// CHECK:             %[[TRANS:.+]] = linalg.transpose ins(%[[TILE]] : tensor<32x8xf32>) outs(%[[EMPTY]] : tensor<8x32xf32>) permutation = [1, 0]
// CHECK:             %[[RESULT:.+]] = tensor.insert_slice %[[TRANS]] into %[[OUT]][0, 0, 0, 0, 0, 0] [1, 1, 1, 1, 8, 32] [1, 1, 1, 1, 1, 1]

// CHECK-ALL:         return %[[RESULT]]

// -----

func.func @simple_pad_and_pack(%input: tensor<5x1xf32>, %output: tensor<1x1x8x2xf32>, %pad: f32) -> tensor<1x1x8x2xf32> {
  %0 = tensor.pack %input padding_value(%pad : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %output : tensor<5x1xf32> -> tensor<1x1x8x2xf32>
  return %0 : tensor<1x1x8x2xf32>
}
// CHECK-ALL-LABEL: func.func @simple_pad_and_pack
// CHECK-ALL-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-ALL-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-ALL-SAME:    %[[PAD_VAL:[A-Za-z0-9]+]]:
// CHECK-ALL:         %[[PAD:.+]] = tensor.pad %[[IN]] low[0, 0] high[3, 1]
// CHECK-ALL:           tensor.yield %[[PAD_VAL]]
// CHECK-ALL:         %[[INSERT:.+]] = tensor.insert_slice %[[PAD]] into %[[OUT]][0, 0, 0, 0] [1, 1, 8, 2] [1, 1, 1, 1]
// CHECK-ALL:         return %[[INSERT]]

// -----

func.func @simple_NC_to_CNnc(%arg0: tensor<32x8xf32>, %arg1: tensor<1x1x32x8xf32>) -> tensor<1x1x32x8xf32>{
  %0 = tensor.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : tensor<32x8xf32> -> tensor<1x1x32x8xf32>
  return %0 : tensor<1x1x32x8xf32>
}
// CHECK-ALL-LABEL: func.func @simple_NC_to_CNnc
// CHECK-ALL-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-ALL-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-ALL:         %[[INSERT:.+]] = tensor.insert_slice %[[IN]] into %[[OUT]][0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK-ALL:         return %[[INSERT]]

// -----

func.func @KCRS_to_KCRSsr(%arg0: tensor<1x1x128x64xf32>, %arg1: tensor<1x1x4x8x8x32xf32>) -> tensor<1x1x4x8x8x32xf32> {
  %0 = tensor.pack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : tensor<1x1x128x64xf32> -> tensor<1x1x4x8x8x32xf32>
  return %0 : tensor<1x1x4x8x8x32xf32>
}
// CHECK-ALL:       func.func @KCRS_to_KCRSsr
// CHECK-ALL-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-ALL-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-ALL:         %[[EXPAND:.+]] = tensor.expand_shape %[[IN]] {{\[}}[0], [1], [2, 3], [4, 5]] output_shape [1, 1, 4, 32, 8, 8] : tensor<1x1x128x64xf32> into tensor<1x1x4x32x8x8xf32>
// CHECK-ALL:          %[[TRANSP:.+]] = linalg.transpose
// CHECK-ALL-SAME:       ins(%[[EXPAND]] : tensor<1x1x4x32x8x8xf32>)
// CHECK-ALL-SAME:       outs(%[[OUT]] : tensor<1x1x4x8x8x32xf32>)
// CHECK-ALL-SAME:       permutation = [0, 1, 2, 4, 5, 3]
// CHECK-ALL:         return %[[TRANSP]]

// -----

func.func @pad_and_pack(%arg0: tensor<13x15xf32>, %arg1: tensor<2x8x8x2xf32>, %arg2: f32) -> tensor<2x8x8x2xf32> {
  %0 = tensor.pack %arg0 padding_value(%arg2 : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %arg1 : tensor<13x15xf32> -> tensor<2x8x8x2xf32>
  return %0 : tensor<2x8x8x2xf32>
}
// CHECK-ALL:       func.func @pad_and_pack
// CHECK-ALL-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-ALL-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-ALL-SAME:    %[[PAD_VAL:[A-Za-z0-9]+]]:
// CHECK-ALL:         %[[PAD:.+]] = tensor.pad %[[IN]] low[0, 0] high[3, 1]
// CHECK-ALL:           tensor.yield %[[PAD_VAL]]
// CHECK-ALL:         } : tensor<13x15xf32> to tensor<16x16xf32>
// CHECK-ALL:         %[[EXPAND:.+]] = tensor.expand_shape %[[PAD]] {{\[}}[0, 1], [2, 3]] output_shape [2, 8, 8, 2] : tensor<16x16xf32> into tensor<2x8x8x2xf32>
// CHECK-ALL:         %[[TRANS:.+]] = linalg.transpose
// CHECK-ALL-SAME:      ins(%[[EXPAND]] : tensor<2x8x8x2xf32>)
// CHECK-ALL-SAME:      outs(%[[OUT:.+]] : tensor<2x8x8x2xf32>)
// CHECK-ALL-SAME:      permutation = [0, 2, 1, 3]
// CHECK-ALL:         return %[[TRANSP]]

// -----

func.func @KC_to_CKck(%arg0: tensor<128x256xf32>, %arg1: tensor<32x4x32x8xf32>) -> tensor<32x4x32x8xf32> {
  %0 = tensor.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : tensor<128x256xf32> -> tensor<32x4x32x8xf32>
  return %0 : tensor<32x4x32x8xf32>
}
// CHECK-ALL:       func.func @KC_to_CKck
// CHECK-ALL-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-ALL-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-ALL:         %[[EXPAND:.+]] = tensor.expand_shape %[[IN]] {{\[}}[0, 1], [2, 3]] output_shape [4, 32, 32, 8] : tensor<128x256xf32> into tensor<4x32x32x8xf32>
// CHECK-ALL:         %[[TRANSP:.+]] = linalg.transpose
// CHECK-ALL-SAME:      ins(%[[EXPAND]] : tensor<4x32x32x8xf32>)
// CHECK-ALL-SAME:      outs(%[[OUT]] : tensor<32x4x32x8xf32>)
// CHECK-ALL-SAME:      permutation = [2, 0, 1, 3]
// CHECK-ALL:         return %[[TRANSP]]

// -----

func.func @simple_KCRSsr_to_KCRS(%arg0: tensor<1x1x1x1x8x32xf32>, %arg1: tensor<1x1x32x8xf32>) -> tensor<1x1x32x8xf32> {
  %0 = tensor.unpack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : tensor<1x1x1x1x8x32xf32> -> tensor<1x1x32x8xf32>
  return %0 : tensor<1x1x32x8xf32>
}
// CHECK-ALL-LABEL: func.func @simple_KCRSsr_to_KCRS
// CHECK-ALL-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-ALL-SAME:    %[[OUT:[A-Za-z0-9]+]]:

// CHECK-RESHAPE:     %[[EMPTY:.+]] = tensor.empty() : tensor<1x1x1x32x1x8xf32>
// CHECK-RESHAPE:     %[[TRANS:.+]] = linalg.transpose ins(%[[IN]] : tensor<1x1x1x1x8x32xf32>) outs(%[[EMPTY]] : tensor<1x1x1x32x1x8xf32>) permutation = [0, 1, 2, 5, 3, 4]
// CHECK-RESHAPE:     %[[COLLAPSE:.+]] = tensor.collapse_shape %[[TRANS]] {{\[}}[0], [1], [2, 3], [4, 5]] : tensor<1x1x1x32x1x8xf32> into tensor<1x1x32x8xf32>
// CHECK-RESHAPE:     %[[RESULT:.+]] = linalg.copy ins(%[[COLLAPSE]] : tensor<1x1x32x8xf32>) outs(%[[OUT]] : tensor<1x1x32x8xf32>) -> tensor<1x1x32x8xf32>

// CHECK:             %[[TILE:.+]] = tensor.extract_slice %[[IN]]
// CHECK-SAME:          [0, 0, 0, 0, 0, 0] [1, 1, 1, 1, 8, 32] [1, 1, 1, 1, 1, 1]
// CHECK:             %[[EMPTY:.+]] = tensor.empty() : tensor<32x8xf32>
// CHECK:             %[[TRANS:.+]] = linalg.transpose ins(%[[TILE]] : tensor<8x32xf32>) outs(%[[EMPTY]] : tensor<32x8xf32>) permutation = [1, 0]
// CHECK:             %[[RESULT:.+]] = tensor.insert_slice %[[TRANS]] into %[[OUT]][0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK-ALL:         return %[[RESULT]]

// -----

func.func @simple_unpack_and_extract_slice(%input: tensor<1x1x8x2xf32>, %output: tensor<5x1xf32>) -> tensor<5x1xf32> {
  %0 = tensor.unpack %input inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %output : tensor<1x1x8x2xf32> -> tensor<5x1xf32>
  return %0 : tensor<5x1xf32>
}
// CHECK-ALL-LABEL: func.func @simple_unpack_and_extract_slice
// CHECK-ALL-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-ALL-SAME:    %[[OUT:[A-Za-z0-9]+]]:

// CHECK:             %[[TILE:.+]] = tensor.extract_slice %[[IN]][0, 0, 0, 0] [1, 1, 8, 2] [1, 1, 1, 1]
// CHECK:             %[[RES:.+]] = tensor.extract_slice %[[TILE]][0, 0] [5, 1] [1, 1]

// CHECK-RESHAPE:     %[[RES:.+]] = tensor.extract_slice %[[IN]][0, 0, 0, 0] [1, 1, 5, 1] [1, 1, 1, 1]

// CHECK-ALL:         return %[[RES:.+]]

// -----

func.func @simple_CNnc_to_NC(%arg0: tensor<1x1x32x8xf32>, %arg1: tensor<32x8xf32>) -> tensor<32x8xf32>{
  %0 = tensor.unpack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : tensor<1x1x32x8xf32> -> tensor<32x8xf32>
  return %0 : tensor<32x8xf32>
}
// CHECK-ALL-LABEL: func.func @simple_CNnc_to_NC
// CHECK-ALL-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-ALL-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-ALL:         %[[TILE:.+]] = tensor.extract_slice %[[IN]][0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK-ALL:         return %[[TILE]]

// -----

func.func @KCRSsr_to_KCRS(%arg0: tensor<13x12x4x8x8x32xf32>, %arg1: tensor<13x12x128x64xf32>) -> tensor<13x12x128x64xf32> {
  %0 = tensor.unpack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : tensor<13x12x4x8x8x32xf32> -> tensor<13x12x128x64xf32>
  return %0 : tensor<13x12x128x64xf32>
}
// CHECK-ALL:       func.func @KCRSsr_to_KCRS
// CHECK-ALL-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-ALL-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-ALL:         %[[EMPTY:.+]] = tensor.empty() : tensor<13x12x4x32x8x8xf32>
// CHECK-ALL:         %[[TRANSP:.+]] = linalg.transpose
// CHECK-ALL-SAME:      ins(%[[IN]] : tensor<13x12x4x8x8x32xf32>)
// CHECK-ALL-SAME:      outs(%[[EMPTY]] : tensor<13x12x4x32x8x8xf32>)
// CHECK-ALL-SAME:      permutation = [0, 1, 2, 5, 3, 4]
// CHECK-ALL:         %[[COLLAPSE:.+]] = tensor.collapse_shape %[[TRANSP]]
// CHECK-ALL-SAME:      {{\[}}[0], [1], [2, 3], [4, 5]] : tensor<13x12x4x32x8x8xf32> into tensor<13x12x128x64xf32>
// CHECK-ALL:         %[[COPY:.]] = linalg.copy ins(%[[COLLAPSE]]
// CHECK-ALL-SAME:        outs(%[[OUT]]
// CHECK-ALL:         return %[[COPY]]

// -----

func.func @unpack_and_extract_slice(%arg0: tensor<2x8x8x2xf32>, %arg1: tensor<13x15xf32>) -> tensor<13x15xf32> {
  %0 = tensor.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %arg1 : tensor<2x8x8x2xf32> -> tensor<13x15xf32>
  return %0 : tensor<13x15xf32>
}
// CHECK-ALL:      func.func @unpack_and_extract_slice
// CHECK-ALL-SAME:   %[[IN:[A-Za-z0-9]+]]:
// CHECK-ALL-SAME:   %[[OUT:[A-Za-z0-9]+]]:
// CHECK-ALL:          %[[EMPTY:.+]] = tensor.empty() : tensor<2x8x8x2xf32>
// CHECK-ALL:          %[[TRANSP:.+]] = linalg.transpose
// CHECK-ALL-SAME:       ins(%[[IN]] : tensor<2x8x8x2xf32>)
// CHECK-ALL-SAME:       outs(%[[EMPTY]] : tensor<2x8x8x2xf32>)
// CHECK-ALL-SAME:       permutation = [0, 2, 1, 3]
// CHECK-ALL:          %[[COLLAPSE:.+]] = tensor.collapse_shape %[[TRANSP]]
// CHECK-ALL-SAME:       {{\[}}[0, 1], [2, 3]] : tensor<2x8x8x2xf32> into tensor<16x16xf32>
// CHECK-ALL:          %[[SLICE:.+]] = tensor.extract_slice %[[COLLAPSE]]
// CHECK-ALL-SAME:       [0, 0] [13, 15] [1, 1] : tensor<16x16xf32> to tensor<13x15xf32>
// CHECK-ALL:          %[[COPY:.]] = linalg.copy ins(%[[SLICE]]
// CHECK-ALL-SAME:         outs(%[[OUT]]
// CHECK-ALL:          return %[[COPY]]
// -----

func.func @CKck_to_KC(%arg0: tensor<32x4x32x8xf32>, %arg1: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %0 = tensor.unpack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : tensor<32x4x32x8xf32> -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}
// CHECK-ALL:      func.func @CKck_to_KC
// CHECK-ALL-SAME:   %[[IN:[A-Za-z0-9]+]]:
// CHECK-ALL-SAME:   %[[OUT:[A-Za-z0-9]+]]:
// CHECK-ALL:        %[[TRANSP:.+]] = linalg.transpose ins(%[[IN]]
// CHECK-ALL:        %[[COLLAPSED:.+]] = tensor.collapse_shape %[[TRANSP]] {{.+}} : tensor<4x32x32x8xf32> into tensor<128x256xf32>
// CHECK-ALL:        %[[RES:.+]] = linalg.copy ins(%[[COLLAPSED]]
// CHECK-ALL:        return %[[RES]]

// -----

func.func @pack_matmul_DYN_LHS(%src: tensor<?x?xf32>, %dest: tensor<?x?x16x1xf32>) -> tensor<?x?x16x1xf32> {
  %pack = tensor.pack %src inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %dest : tensor<?x?xf32> -> tensor<?x?x16x1xf32>
  return %pack : tensor<?x?x16x1xf32>
}
// CHECK-ALL-DAG:  #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s0 * 16 - s1)>
// CHECK-ALL-DAG:  #[[MAP1:.+]] = affine_map<()[s0, s1] -> (s0 - s1)>
// CHECK-ALL:      func.func @pack_matmul_DYN_LHS
// CHECK-ALL-SAME:   %[[IN:[A-Za-z0-9]+]]:
// CHECK-ALL-SAME:   %[[OUT:[A-Za-z0-9]+]]:
// CHECK-ALL-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-ALL-DAG:    %[[D0:.+]] = tensor.dim %[[IN]], %c0 : tensor<?x?xf32>
// CHECK-ALL-DAG:    %[[H0:.+]] = affine.apply #[[MAP0]]
// CHECK-ALL-DAG:    %[[H1:.+]] = affine.apply #[[MAP1]]
// CHECK-ALL:        %[[PAD:.+]] = tensor.pad %[[IN]] low[0, 0] high[%[[H0]], %[[H1]]]
// CHECK-ALL:        %[[EXPANDED:.+]] = tensor.expand_shape %[[PAD]]
// CHECK-ALL-SAME:     {{\[}}[0, 1], [2, 3]]
// CHECK-ALL:        %[[TRANSP:.+]] = linalg.transpose
// CHECK-ALL-SAME:     ins(%[[EXPANDED]] : tensor<?x16x?x1xf32>)
// CHECK-ALL-SAME:     outs(%[[OUT]] : tensor<?x?x16x1xf32>)
// CHECK-ALL-SAME:   permutation = [0, 2, 1, 3]
// CHECK-ALL:        return %[[TRANSP]]

// -----

func.func @pack_matmul_DYN_RHS(%src: tensor<?x?xf32>, %dest: tensor<?x?x16x1xf32>) -> tensor<?x?x16x1xf32> {
  %pack = tensor.pack %src outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 1] into %dest : tensor<?x?xf32> -> tensor<?x?x16x1xf32>
  return %pack : tensor<?x?x16x1xf32>
}
// CHECK-ALL-DAG:  #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s0 * 16 - s1)>
// CHECK-ALL-DAG:  #[[MAP1:.+]] = affine_map<()[s0, s1] -> (s0 - s1)>
// CHECK-ALL:      func.func @pack_matmul_DYN_RHS
// CHECK-ALL-SAME:   %[[IN:[A-Za-z0-9]+]]:
// CHECK-ALL-SAME:   %[[OUT:[A-Za-z0-9]+]]:
// CHECK-ALL-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-ALL-DAG:    %[[H0:.+]] = affine.apply #[[MAP1]]
// CHECK-ALL-DAG:    %[[H1:.+]] = affine.apply #[[MAP0]]
// CHECK-ALL:        %[[PAD:.+]] = tensor.pad %[[IN]] low[0, 0] high[%[[H0]], %[[H1]]]
// CHECK-ALL:        %[[EXPANDED:.+]] = tensor.expand_shape %[[PAD]]
// CHECK-ALL-SAME:     {{\[}}[0, 1], [2, 3]]
// CHECK-ALL:        %[[TRANSP:.+]] = linalg.transpose
// CHECK-ALL-SAME:     ins(%[[EXPANDED]] : tensor<?x1x?x16xf32>)
// CHECK-ALL-SAME:     outs(%[[OUT]] : tensor<?x?x16x1xf32>)
// CHECK-ALL-SAME:     permutation = [2, 0, 3, 1]
