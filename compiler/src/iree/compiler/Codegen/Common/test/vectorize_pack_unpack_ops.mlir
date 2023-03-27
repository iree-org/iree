// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-decompose-pack-unpack-ops, iree-codegen-vectorize-pack-unpack-ops))" --split-input-file %s | FileCheck %s

func.func @simple_KCRS_to_KCRSsr(%arg0: tensor<1x1x32x8xf32>, %arg1: tensor<1x1x1x1x8x32xf32>) -> tensor<1x1x1x1x8x32xf32> {
  %0 = tensor.pack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : tensor<1x1x32x8xf32> -> tensor<1x1x1x1x8x32xf32>
  return %0 : tensor<1x1x1x1x8x32xf32>
}
// CHECK-LABEL: func.func @simple_KCRS_to_KCRSsr
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[READ:.+]] = vector.transfer_read %[[IN]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]]
// CHECK-SAME:      {in_bounds = [true, true]
// CHECK-SAME:    : tensor<1x1x32x8xf32>, vector<32x8xf32>
// CHECK:         %[[TRANS:.+]] =  vector.transpose %[[READ]], [1, 0] : vector<32x8xf32> to vector<8x32xf32>
// CHECK:         %[[WRITE:.+]] = vector.transfer_write %[[TRANS]],
// CHECK-SAME:      %[[OUT]][%[[C0]], %[[C0]], %[[C0]], %[[C0]], %[[C0]], %[[C0]]]
// CHECK-SAME:      {in_bounds = [true, true]} : vector<8x32xf32>, tensor<1x1x1x1x8x32xf32>

// -----

func.func @simple_pad_and_pack(%input: tensor<5x1xf32>, %output: tensor<1x1x8x2xf32>, %pad: f32) -> tensor<1x1x8x2xf32> {
  %0 = tensor.pack %input padding_value(%pad : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %output : tensor<5x1xf32> -> tensor<1x1x8x2xf32>
  return %0 : tensor<1x1x8x2xf32>
}
// CHECK-LABEL: func.func @simple_pad_and_pack
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[PAD:[A-Za-z0-9]+]]:
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[READ:.+]] = vector.transfer_read %[[IN]][%[[C0]], %[[C0]]], %[[PAD]]
// CHECK-SAME:      : tensor<5x1xf32>, vector<8x2xf32>
// CHECK:         %[[WRITE:.+]] = vector.transfer_write %[[READ]],
// CHECK-SAME:      %[[OUT]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]]
// CHECK-SAME:     {in_bounds = [true, true]} : vector<8x2xf32>, tensor<1x1x8x2xf32>

// -----

func.func @simple_NC_to_CNnc(%arg0: tensor<32x8xf32>, %arg1: tensor<1x1x32x8xf32>) -> tensor<1x1x32x8xf32>{
  %0 = tensor.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : tensor<32x8xf32> -> tensor<1x1x32x8xf32>
  return %0 : tensor<1x1x32x8xf32>
}
// CHECK-LABEL: func.func @simple_NC_to_CNnc
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[READ:.+]] = vector.transfer_read %[[IN]][%[[C0]], %[[C0]]]
// CHECK-SAME:      {in_bounds = [true, true]}
// CHECK-SAME:    : tensor<32x8xf32>, vector<32x8xf32>
// CHECK:         %[[WRITE:.+]] = vector.transfer_write %[[READ]],
// CHECK-SAME:      %[[OUT]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]]
// CHECK-SAME:      {in_bounds = [true, true]} : vector<32x8xf32>, tensor<1x1x32x8xf32>

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
// CHECK-DAG:     %[[CST:.+]] = arith.constant 0.000000e+00 : f32
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
// CHECK:             %[[ITER_SLICE:.+]] = tensor.extract_slice %[[ITER1]]
// CHECK-SAME:          [0, 0, %[[I]], %[[J]], 0, 0] [1, 1, 1, 1, 8, 32] [1, 1, 1, 1, 1, 1]
// CHECK-SAME:          tensor<1x1x4x8x8x32xf32> to tensor<1x1x1x1x8x32xf32>
// CHECK:             %[[READ:.+]] = vector.transfer_read %[[IN]]
// CHECK-SAME:          [%[[C0]], %[[C0]], %[[IDX2]], %[[IDX3]]]
// CHECK-SAME:          {in_bounds = [true, true]}
// CHECK-SAME:          : tensor<1x1x128x64xf32>, vector<32x8xf32>
// CHECK:             %[[TRANS:.+]] = vector.transpose %[[READ]], [1, 0] : vector<32x8xf32> to vector<8x32xf32>
// CHECK:             %[[WRITE:.+]] = vector.transfer_write %[[TRANS]]
// CHECK-SAME:          %[[ITER_SLICE]][%[[C0]], %[[C0]], %[[C0]], %[[C0]], %[[C0]], %[[C0]]
// CHECK-SAME:          {in_bounds = [true, true]}
// CHECK-SAME:          : vector<8x32xf32>, tensor<1x1x1x1x8x32xf32>
// CHECK:             %[[INSERT:.+]] = tensor.insert_slice %[[WRITE]] into %[[ITER1]]
// CHECK-SAME:          [0, 0, %[[I]], %[[J]], 0, 0]
// CHECK:             scf.yield %[[INSERT]]
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
// CHECK:       func.func @pad_and_pack
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[PAD:[A-Za-z0-9]+]]:
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
// CHECK:             %[[SLICE:.+]] = tensor.extract_slice %[[IN]][%[[IDX0]], %[[IDX1]]] [%[[SZ0]], %[[SZ1]]]
// CHECK:             %[[ITER_SLICE:.+]] = tensor.extract_slice %[[ITER1]]
// CHECK-SAME:          [%[[I]], %[[J]], 0, 0] [1, 1, 8, 2] [1, 1, 1, 1]
// CHECK:             %[[READ:.+]] = vector.transfer_read %[[SLICE]]
// CHECK-SAME:          [%[[C0]], %[[C0]]], %[[PAD]]
// CHECK-SAME:          : tensor<?x?xf32>, vector<8x2xf32>
// CHECK:             %[[WRITE:.+]] = vector.transfer_write %[[READ]]
// CHECK-SAME:          %[[ITER_SLICE]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]
// CHECK-SAME:          {in_bounds = [true, true]}
// CHECK-SAME:          : vector<8x2xf32>, tensor<1x1x8x2xf32>
// CHECK:             %[[INSERT:.+]] = tensor.insert_slice %[[WRITE]] into %[[ITER1]]
// CHECK-SAME:          [%[[I]], %[[J]], 0, 0]
// CHECK:             scf.yield %[[INSERT]]
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
// CHECK-DAG:     %[[CST:.+]] = arith.constant 0.000000e+00 : f32
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
// CHECK:             %[[ITER_SLICE:.+]] = tensor.extract_slice %[[ITER1]]
// CHECK-SAME:          [%[[C]], %[[K]], 0, 0] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK:             %[[READ:.+]] = vector.transfer_read %[[IN]]
// CHECK-SAME:          [%[[IN_K]], %[[IN_C]]]
// CHECK-SAME:          {in_bounds = [true, true]}
// CHECK-SAME:          : tensor<128x256xf32>, vector<32x8xf32>
// CHECK:             %[[WRITE:.+]] = vector.transfer_write %[[READ]]
// CHECK-SAME:          %[[ITER_SLICE]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]
// CHECK-SAME:          {in_bounds = [true, true]}
// CHECK-SAME:          : vector<32x8xf32>, tensor<1x1x32x8xf32>
// CHECK:             %[[INSERT:.+]] = tensor.insert_slice %[[WRITE]] into %[[ITER1]]
// CHECK-SAME:          [%[[C]], %[[K]], 0, 0] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK:             scf.yield %[[INSERT]]
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
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[READ:.+]] = vector.transfer_read %[[IN]]
// CHECK-SAME:      [%[[C0]], %[[C0]], %[[C0]], %[[C0]], %[[C0]], %[[C0]]], %[[ZERO]]
// CHECK-SAME:      {in_bounds = [true, true]} : tensor<1x1x1x1x8x32xf32>, vector<8x32xf32>
// CHECK:         %[[TRANSP:.+]] = vector.transpose %[[READ]], [1, 0]
// CHECK:         %[[WRITE:.+]] = vector.transfer_write %[[TRANSP]]
// CHECK-SAME:      %[[OUT]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]]
// CHECK-SAME:      {in_bounds = [true, true]} : vector<32x8xf32>, tensor<1x1x32x8xf32>
// CHECK:         return %[[WRITE]]

// -----

func.func @simple_unpack_and_extract_slice(%input: tensor<1x1x8x2xf32>, %output: tensor<5x1xf32>) -> tensor<5x1xf32> {
  %0 = tensor.unpack %input inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %output : tensor<1x1x8x2xf32> -> tensor<5x1xf32>
  return %0 : tensor<5x1xf32>
}
// CHECK-LABEL: func.func @simple_unpack_and_extract_slice
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %[[EMPTY:.+]] = tensor.empty() : tensor<8x2xf32>
// CHECK:         %[[READ:.+]] = vector.transfer_read %[[IN]]
// CHECK-SAME:      [%[[C0]], %[[C0]], %[[C0]], %[[C0]]], %[[ZERO]]
// CHECK-SAME:      {in_bounds = [true, true]} : tensor<1x1x8x2xf32>, vector<8x2xf32>
// CHECK:         %[[WRITE:.+]] = vector.transfer_write %[[READ]],
// CHECK-SAME:      %[[EMPTY]][%[[C0]], %[[C0]]]
// CHECK-SAME:      {in_bounds = [true, true]} : vector<8x2xf32>, tensor<8x2xf32>
// CHECK:         %[[RES:.+]] = tensor.extract_slice %[[WRITE]]
// CHECK-SAME:      [0, 0] [5, 1] [1, 1] : tensor<8x2xf32> to tensor<5x1xf32>
// CHECK:         return %[[RES:.+]]

// -----

func.func @simple_CNnc_to_NC(%arg0: tensor<1x1x32x8xf32>, %arg1: tensor<32x8xf32>) -> tensor<32x8xf32>{
  %0 = tensor.unpack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : tensor<1x1x32x8xf32> -> tensor<32x8xf32>
  return %0 : tensor<32x8xf32>
}
// CHECK-LABEL: func.func @simple_CNnc_to_NC
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %[[EMPTY:.+]] = tensor.empty() : tensor<32x8xf32>
// CHECK:         %[[READ:.+]] = vector.transfer_read %[[IN]]
// CHECK-SAME:      [%[[C0]], %[[C0]], %[[C0]], %[[C0]]], %[[ZERO]]
// CHECK-SAME:      {in_bounds = [true, true]} : tensor<1x1x32x8xf32>, vector<32x8xf32>
// CHECK:         %[[WRITE:.+]] = vector.transfer_write %[[READ]],
// CHECK-SAME:      %[[EMPTY]][%[[C0]], %[[C0]]]
// CHECK-SAME:      {in_bounds = [true, true]} : vector<32x8xf32>, tensor<32x8xf32>
// CHECK:         return %[[WRITE]]

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
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
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
// CHECK-DAG:         %[[ITER3_SLICE:.+]] = tensor.extract_slice %[[ITER3]]
// CHECK-SAME:          [%[[K]], %[[C]], %[[R]], %[[S]]] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK:             %[[READ:.+]] = vector.transfer_read %[[IN]]
// CHECK-SAME:          [%[[K]], %[[C]], %[[IN_R]], %[[IN_S]], %[[C0]], %[[C0]]], %[[ZERO]]
// CHECK-SAME:          {in_bounds = [true, true]} : tensor<13x12x4x8x8x32xf32>, vector<8x32xf32>
// CHECK:             %[[TRANSP:.+]] = vector.transpose %[[READ]], [1, 0]
// CHECK:             %[[WRITE:.+]] = vector.transfer_write %[[TRANSP]]
// CHECK-SAME:          %[[ITER3_SLICE]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]]
// CHECK-SAME:          {in_bounds = [true, true]} : vector<32x8xf32>, tensor<1x1x32x8xf32>
// CHECK:             %[[INSERT:.+]] = tensor.insert_slice %[[WRITE]]
// CHECK-SAME:          into %[[ITER3]][%[[K]], %[[C]], %[[R]], %[[S]]] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK:             scf.yield %[[INSERT]]
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
// CHECK-DAG:    %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[RES0:.+]] = scf.for %[[I:.+]] = %[[C0]] to %[[C13]] step %[[C8]]
// CHECK-SAME:     iter_args(%[[ITER0:.+]] = %[[OUT]])
// CHECK-DAG:      %[[OUT_I_SZ:.+]] = affine.min #[[MAP0]](%[[I]])
// CHECK:          %[[RES1:.+]] = scf.for %[[J:.+]] = %[[C0]] to %[[C15]] step %[[C2]]
// CHECK-SAME:       iter_args(%[[ITER1:.+]] = %[[ITER0]])
// CHECK-DAG:        %[[OUT_J_SZ:.+]] = affine.min #[[MAP1]](%[[J]])
// CHECK-DAG:        %[[IN_I:.+]] = affine.apply #[[MAP2]](%[[I]])
// CHECK-DAG:        %[[IN_J:.+]] = affine.apply #[[MAP3]](%[[J]])
// CHECK-DAG:        %[[ITER1_SLICE1:.+]] = tensor.extract_slice %[[ITER1]]
// CHECK-SAME:         [%[[I]], %[[J]]] [%[[OUT_I_SZ]], %[[OUT_J_SZ]]] [1, 1]
// CHECK-DAG:        %[[READ:.+]] = vector.transfer_read %[[IN]]
// CHECK-SAME:         [%[[IN_I]], %[[IN_J]], %[[C0]], %[[C0]]], %[[ZERO]]
// CHECK-SAME:         {in_bounds = [true, true]} : tensor<2x8x8x2xf32>, vector<8x2xf32>
// CHECK-DAG:        %[[ITER1_SLICE2:.+]] = tensor.extract_slice %[[ITER1_SLICE1]]
// CHECK-SAME:         [0, 0] [%[[OUT_I_SZ]], %[[OUT_J_SZ]]] [1, 1]
// CHECK:            %[[WRITE:.+]] = vector.transfer_write %[[READ]]
// CHECK-SAME:         %[[ITER1_SLICE2]][%[[C0]], %[[C0]]]
// CHECK:            %[[INSERT1:.+]] = tensor.insert_slice %[[WRITE]]
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
// CHECK-DAG:    %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[RES0:.+]] = scf.for %[[K:.+]] = %[[C0]] to %[[C128]] step %[[C32]]
// CHECK-SAME:     iter_args(%[[ITER0:.+]] = %[[OUT]])
// CHECK:          %[[RES1:.+]] = scf.for %[[C:.+]] = %[[C0]] to %[[C256]] step %[[C8]]
// CHECK-SAME:       iter_args(%[[ITER1:.+]] = %[[ITER0]])
// CHECK-DAG:        %[[IN_K:.+]] = affine.apply #[[MAP0]](%[[K]])
// CHECK-DAG:        %[[IN_C:.+]] = affine.apply #[[MAP1]](%[[C]])
// CHECK-DAG:        %[[READ:.+]] = vector.transfer_read %[[IN]]
// CHECK-SAME:         [%[[IN_C]], %[[IN_K]], %[[C0]], %[[C0]]], %[[ZERO]]
// CHECK-SAME:         {in_bounds = [true, true]} : tensor<32x4x32x8xf32>, vector<32x8xf32>
// CHECK:            %[[WRITE:.+]] = vector.transfer_write %[[READ]]
// CHECK-SAME:         %[[ITER1]][%[[K]], %[[C]]]
// CHECK:            scf.yield %[[WRITE]]
// CHECK:          }
// CHECK:          scf.yield %[[RES1]]
// CHECK:        }
// CHECK:        return %[[RES0]]
