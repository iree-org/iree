// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-generic-vectorization))" --split-input-file %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-generic-vectorization{enable-vector-masking=true}))" --split-input-file %s | FileCheck %s -check-prefix=CHECK-MASK
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-generic-vectorization{fold-cast-into-contract=true}))" --split-input-file %s | FileCheck %s -check-prefix=CHECK-FOLD

func.func @matmul(%lhs: tensor<3x4xf16>, %rhs: tensor<4x5xf16>, %acc: tensor<3x5xf32>) -> tensor<3x5xf32> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<3x4xf16>, tensor<4x5xf16>) outs(%acc: tensor<3x5xf32>) -> tensor<3x5xf32>
  return %result: tensor<3x5xf32>
}
// CHECK-LABEL: func.func @matmul
// CHECK-SAME:    %[[LHS:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[RHS:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[OUT:[a-zA-Z0-9]+]]
// CHECK:         %[[LHS_VEC:.+]] = vector.transfer_read %[[LHS]]
// CHECK:         %[[RHS_VEC:.+]] = vector.transfer_read %[[RHS]]
// CHECK:         %[[OUT_VEC:.+]] = vector.transfer_read %[[OUT]]
// CHECK:         %[[EXT_LHS:.+]] = arith.extf %[[LHS_VEC]]
// CHECK:         %[[EXT_RHS:.+]] = arith.extf %[[RHS_VEC]]
// CHECK:         %[[RES:.+]] = vector.contract {{.+}} %[[EXT_LHS]], %[[EXT_RHS]], %[[OUT_VEC]]

// CHECK-FOLD-LABEL: func.func @matmul
// CHECK-FOLD-SAME:    %[[LHS:[a-zA-Z0-9]+]]
// CHECK-FOLD-SAME:    %[[RHS:[a-zA-Z0-9]+]]
// CHECK-FOLD-SAME:    %[[OUT:[a-zA-Z0-9]+]]
// CHECK-FOLD:         %[[LHS_VEC:.+]] = vector.transfer_read %[[LHS]]
// CHECK-FOLD:         %[[RHS_VEC:.+]] = vector.transfer_read %[[RHS]]
// CHECK-FOLD:         %[[OUT_VEC:.+]] = vector.transfer_read %[[OUT]]
// CHECK-FOLD:         %[[RES:.+]] = vector.contract {{.+}} %[[LHS_VEC]], %[[RHS_VEC]], %[[OUT_VEC]]

// -----

#map = affine_map<(d0) -> (-d0 + 13, 2)>
#map1 = affine_map<(d0) -> (-d0 + 51, 4)>
#map2 = affine_map<(d0) -> (d0 * 2)>
#map3 = affine_map<(d0, d1) -> (d1 * -2 + 101, d0 * 2)>
#map4 = affine_map<(d0) -> (d0 * 16)>
#map5 = affine_map<(d0, d1) -> (d1 * -16 + 201, d0 * 16)>
func.func @single_static_pack_infer_vector_size(%arg0: tensor<101x201xi8>, %arg1: tensor<13x51x16x2xi8>) -> tensor<13x51x16x2xi8> {
  %c4 = arith.constant 4 : index
  %c51 = arith.constant 51 : index
  %c0_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c13 = arith.constant 13 : index
  %c2 = arith.constant 2 : index
  %0 = scf.for %arg2 = %c0 to %c13 step %c2 iter_args(%arg3 = %arg1) -> (tensor<13x51x16x2xi8>) {
    %1 = scf.for %arg4 = %c0 to %c51 step %c4 iter_args(%arg5 = %arg3) -> (tensor<13x51x16x2xi8>) {
      %2 = affine.min #map(%arg2)
      %3 = affine.min #map1(%arg4)
      %4 = affine.apply #map2(%arg4)
      %5 = affine.min #map3(%3, %arg4)
      %6 = affine.apply #map4(%arg2)
      %7 = affine.min #map5(%2, %arg2)
      %extracted_slice = tensor.extract_slice %arg0[%4, %6] [%5, %7] [1, 1] : tensor<101x201xi8> to tensor<?x?xi8>
      %extracted_slice_0 = tensor.extract_slice %arg5[%arg2, %arg4, 0, 0] [%2, %3, 16, 2] [1, 1, 1, 1] : tensor<13x51x16x2xi8> to tensor<?x?x16x2xi8>
      %pack = tensor.pack %extracted_slice padding_value(%c0_i8 : i8) outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 2] into %extracted_slice_0 : tensor<?x?xi8> -> tensor<?x?x16x2xi8>
      %inserted_slice = tensor.insert_slice %pack into %arg5[%arg2, %arg4, 0, 0] [%2, %3, 16, 2] [1, 1, 1, 1] : tensor<?x?x16x2xi8> into tensor<13x51x16x2xi8>
      scf.yield %inserted_slice : tensor<13x51x16x2xi8>
    }
    scf.yield %1 : tensor<13x51x16x2xi8>
  }
  return %0 : tensor<13x51x16x2xi8>
}
// Direct tensor.pack vectorization is only available with masking.
// TODO: Support non-masking path.
// CHECK-LABEL: func.func @single_static_pack_infer_vector_size
// CHECK:         tensor.pack

// CHECK-MASK: #[[$MAP0:.+]] = affine_map<(d0) -> (-d0 + 13, 2)>
// CHECK-MASK: #[[$MAP1:.+]] = affine_map<(d0) -> (-d0 + 51, 4)>
// CHECK-MASK: #[[$MAP2:.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-MASK: #[[$MAP3:.+]] = affine_map<(d0, d1) -> (d1 * -2 + 101, d0 * 2)>
// CHECK-MASK: #[[$MAP4:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-MASK: #[[$MAP5:.+]] = affine_map<(d0, d1) -> (d1 * -16 + 201, d0 * 16)>
// CHECK-MASK-LABEL: func.func @single_static_pack_infer_vector_size
// CHECK-MASK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-MASK:         %[[C0:.+]] = arith.constant 0 : i8
// CHECK-MASK:         scf.for
// CHECK-MASK:           scf.for
// CHECK-MASK:             %[[WRITE_SZ0:.+]] = affine.min #[[$MAP0]]
// CHECK-MASK:             %[[WRITE_SZ1:.+]] = affine.min #[[$MAP1]]
// CHECK-MASK:             %[[READ_SZ0:.+]] = affine.min #[[$MAP3]]
// CHECK-MASK:             %[[READ_SZ1:.+]] = affine.min #[[$MAP5]]
// CHECK-MASK:             %[[SLICE:.+]] = tensor.extract_slice %[[SRC]][{{.+}}] [%[[READ_SZ0]], %[[READ_SZ1]]]
// CHECK-MASK:             %[[READ_MASK:.+]] = vector.create_mask %[[READ_SZ0]], %[[READ_SZ1]] : vector<8x32xi1>
// CHECK-MASK:             %[[READ:.+]] = vector.transfer_read %[[SLICE]][%{{.+}}], %[[C0]], %[[READ_MASK]]
// CHECK-MASK:             %[[CAST:.+]] = vector.shape_cast %[[READ]] : vector<8x32xi8> to vector<4x2x2x16xi8>
// CHECK-MASK:             %[[TRANSP:.+]] = vector.transpose %[[CAST]], [2, 0, 3, 1]
// CHECK-MASK:             %[[EMPTY:.+]] = tensor.empty(%[[WRITE_SZ0]], %[[WRITE_SZ1]]) : tensor<?x?x16x2xi8>
// CHECK-MASK:             %[[WRITE_MASK:.+]] = vector.create_mask %[[WRITE_SZ0]], %[[WRITE_SZ1]], {{.+}} : vector<2x4x16x2xi1>
// CHECK-MASK:             vector.transfer_write %[[TRANSP]], %[[EMPTY]][{{.+}}, %[[WRITE_MASK]]


// -----

#map = affine_map<(d0)[s0] -> (-d0 + s0, 2)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 4)>
#map2 = affine_map<(d0) -> (d0 * 2)>
#map3 = affine_map<(d0, d1)[s0] -> (d1 * -2 + s0, d0 * 2)>
#map4 = affine_map<(d0) -> (d0 * 16)>
#map5 = affine_map<(d0, d1)[s0] -> (d1 * -16 + s0, d0 * 16)>
func.func @single_dynamic_pack_infer_vector_size(%arg0: tensor<?x?xi8>, %arg1: tensor<?x?x16x2xi8>) -> tensor<?x?x16x2xi8> {
  %c4 = arith.constant 4 : index
  %c0_i8 = arith.constant 0 : i8
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg1, %c0 : tensor<?x?x16x2xi8>
  %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?x16x2xi8>
  %0 = scf.for %arg2 = %c0 to %dim step %c2 iter_args(%arg3 = %arg1) -> (tensor<?x?x16x2xi8>) {
    %1 = scf.for %arg4 = %c0 to %dim_0 step %c4 iter_args(%arg5 = %arg3) -> (tensor<?x?x16x2xi8>) {
      %2 = affine.min #map(%arg2)[%dim]
      %3 = affine.min #map1(%arg4)[%dim_0]
      %dim_1 = tensor.dim %arg0, %c0 : tensor<?x?xi8>
      %dim_2 = tensor.dim %arg0, %c1 : tensor<?x?xi8>
      %4 = affine.apply #map2(%arg4)
      %5 = affine.min #map3(%3, %arg4)[%dim_1]
      %6 = affine.apply #map4(%arg2)
      %7 = affine.min #map5(%2, %arg2)[%dim_2]
      %extracted_slice = tensor.extract_slice %arg0[%4, %6] [%5, %7] [1, 1] : tensor<?x?xi8> to tensor<?x?xi8>
      %extracted_slice_3 = tensor.extract_slice %arg5[%arg2, %arg4, 0, 0] [%2, %3, 16, 2] [1, 1, 1, 1] : tensor<?x?x16x2xi8> to tensor<?x?x16x2xi8>
      %pack = tensor.pack %extracted_slice padding_value(%c0_i8 : i8) outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 2] into %extracted_slice_3 : tensor<?x?xi8> -> tensor<?x?x16x2xi8>
      %inserted_slice = tensor.insert_slice %pack into %arg5[%arg2, %arg4, 0, 0] [%2, %3, 16, 2] [1, 1, 1, 1] : tensor<?x?x16x2xi8> into tensor<?x?x16x2xi8>
      scf.yield %inserted_slice : tensor<?x?x16x2xi8>
    }
    scf.yield %1 : tensor<?x?x16x2xi8>
  }
  return %0 : tensor<?x?x16x2xi8>
}
// Direct tensor.pack vectorization is only available with masking.
// TODO: Support non-masking path.
// CHECK-LABEL: func.func @single_dynamic_pack_infer_vector_size
// CHECK:         tensor.pack

// CHECK-MASK: #[[$MAP0:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 2)>
// CHECK-MASK: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 4)>
// CHECK-MASK: #[[$MAP2:.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-MASK: #[[$MAP3:.+]] = affine_map<(d0, d1)[s0] -> (d1 * -2 + s0, d0 * 2)>
// CHECK-MASK: #[[$MAP4:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-MASK: #[[$MAP5:.+]] = affine_map<(d0, d1)[s0] -> (d1 * -16 + s0, d0 * 16)>
// CHECK-MASK-LABEL: func.func @single_dynamic_pack_infer_vector_size
// CHECK-MASK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-MASK:         %[[C0:.+]] = arith.constant 0 : i8
// CHECK-MASK:         scf.for
// CHECK-MASK:           scf.for
// CHECK-MASK:             %[[WRITE_SZ0:.+]] = affine.min #[[$MAP0]]
// CHECK-MASK:             %[[WRITE_SZ1:.+]] = affine.min #[[$MAP1]]
// CHECK-MASK:             %[[READ_SZ0:.+]] = affine.min #[[$MAP3]]
// CHECK-MASK:             %[[READ_SZ1:.+]] = affine.min #[[$MAP5]]
// CHECK-MASK:             %[[SLICE:.+]] = tensor.extract_slice %[[SRC]][{{.+}}] [%[[READ_SZ0]], %[[READ_SZ1]]]
// CHECK-MASK:             %[[READ_MASK:.+]] = vector.create_mask %[[READ_SZ0]], %[[READ_SZ1]] : vector<8x32xi1>
// CHECK-MASK:             %[[READ:.+]] = vector.transfer_read %[[SLICE]][%{{.+}}], %[[C0]], %[[READ_MASK]]
// CHECK-MASK:             %[[CAST:.+]] = vector.shape_cast %[[READ]] : vector<8x32xi8> to vector<4x2x2x16xi8>
// CHECK-MASK:             %[[TRANSP:.+]] = vector.transpose %[[CAST]], [2, 0, 3, 1]
// CHECK-MASK:             %[[EMPTY:.+]] = tensor.empty(%[[WRITE_SZ0]], %[[WRITE_SZ1]]) : tensor<?x?x16x2xi8>
// CHECK-MASK:             %[[WRITE_MASK:.+]] = vector.create_mask %[[WRITE_SZ0]], %[[WRITE_SZ1]], {{.+}} : vector<2x4x16x2xi1>
// CHECK-MASK:             vector.transfer_write %[[TRANSP]], %[[EMPTY]][{{.+}}, %[[WRITE_MASK]]

// -----

#map = affine_map<()[s0] -> (s0 ceildiv 16)>
#map1 = affine_map<(d0)[s0] -> (4, -d0 + s0 ceildiv 16)>
#map2 = affine_map<(d0) -> (-d0 + 64, 6)>
#map3 = affine_map<(d0, d1) -> (d1 * -2 + 128, d0 * 2)>
#map4 = affine_map<(d0, d1)[s0] -> (d1 * -16 + s0, d0 * 16)>
#map5 = affine_map<(d0) -> (d0 * 16)>
#map6 = affine_map<(d0) -> (d0 * 2)>
#map7 = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @generic_pack_infer_vector_size(%arg0: tensor<?x32x128xf32>) -> tensor<32x?x64x16x2xbf16> {
  %c6 = arith.constant 6 : index
  %c64 = arith.constant 64 : index
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : bf16
  %dim = tensor.dim %arg0, %c0 : tensor<?x32x128xf32>
  %0 = affine.apply #map()[%dim]
  %1 = tensor.empty(%dim) : tensor<32x128x?xbf16>
  %2 = tensor.empty(%0) : tensor<32x?x64x16x2xbf16>
  %3 = scf.for %arg1 = %c0 to %c32 step %c2 iter_args(%arg2 = %2) -> (tensor<32x?x64x16x2xbf16>) {
    %4 = scf.for %arg3 = %c0 to %0 step %c4 iter_args(%arg4 = %arg2) -> (tensor<32x?x64x16x2xbf16>) {
      %5 = scf.for %arg5 = %c0 to %c64 step %c6 iter_args(%arg6 = %arg4) -> (tensor<32x?x64x16x2xbf16>) {
        %6 = affine.min #map1(%arg3)[%dim]
        %7 = affine.min #map2(%arg5)
        %8 = affine.min #map3(%7, %arg5)
        %9 = affine.min #map4(%6, %arg3)[%dim]
        %10 = affine.apply #map5(%arg3)
        %11 = affine.apply #map6(%arg5)
        %extracted_slice = tensor.extract_slice %1[%arg1, %11, %10] [2, %8, %9] [1, 1, 1] : tensor<32x128x?xbf16> to tensor<2x?x?xbf16>
        %extracted_slice_0 = tensor.extract_slice %arg0[%10, %arg1, %11] [%9, 2, %8] [1, 1, 1] : tensor<?x32x128xf32> to tensor<?x2x?xf32>
        %12 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_0 : tensor<?x2x?xf32>) outs(%extracted_slice : tensor<2x?x?xbf16>) {
        ^bb0(%in: f32, %out: bf16):
          %13 = arith.truncf %in : f32 to bf16
          linalg.yield %13 : bf16
        } -> tensor<2x?x?xbf16>
        %extracted_slice_1 = tensor.extract_slice %arg6[%arg1, %arg3, %arg5, 0, 0] [2, %6, %7, 16, 2] [1, 1, 1, 1, 1] : tensor<32x?x64x16x2xbf16> to tensor<2x?x?x16x2xbf16>
        %pack = tensor.pack %12 padding_value(%cst : bf16) outer_dims_perm = [0, 2, 1] inner_dims_pos = [2, 1] inner_tiles = [16, 2] into %extracted_slice_1 : tensor<2x?x?xbf16> -> tensor<2x?x?x16x2xbf16>
        %inserted_slice = tensor.insert_slice %pack into %arg6[%arg1, %arg3, %arg5, 0, 0] [2, %6, %7, 16, 2] [1, 1, 1, 1, 1] : tensor<2x?x?x16x2xbf16> into tensor<32x?x64x16x2xbf16>
        scf.yield %inserted_slice : tensor<32x?x64x16x2xbf16>
      }
      scf.yield %5 : tensor<32x?x64x16x2xbf16>
    }
    scf.yield %4 : tensor<32x?x64x16x2xbf16>
  }
  return %3 : tensor<32x?x64x16x2xbf16>
}
// CHECK-MASK: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
// CHECK-MASK: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (4, -d0 + s0 ceildiv 16)>
// CHECK-MASK: #[[$MAP2:.+]] = affine_map<(d0) -> (-d0 + 64, 6)>
// CHECK-MASK: #[[$MAP3:.+]] = affine_map<(d0, d1) -> (d1 * -2 + 128, d0 * 2)>
// CHECK-MASK: #[[$MAP4:.+]] = affine_map<(d0, d1)[s0] -> (d1 * -16 + s0, d0 * 16)>
// CHECK-MASK: #[[$MAP5:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-MASK: #[[$MAP6:.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-MASK-LABEL: func.func @generic_pack_infer_vector_size
// CHECK-MASK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-MASK-DAG:     %[[C0_BF16:.+]] = arith.constant 0.000000e+00 : bf16
// CHECK-MASK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-MASK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-MASK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-MASK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK-MASK:         %[[D0:.+]] = tensor.dim %[[SRC]], %[[C0]] : tensor<?x32x128xf32>
// CHECK-MASK:         %[[GENERIC_EMPTY:.+]] = tensor.empty(%[[D0]]) : tensor<32x128x?xbf16>
// CHECK-MASK:         scf.for
// CHECK-MASK:         scf.for
// CHECK-MASK:         scf.for
// CHECK-MASK-SAME:      iter_args(%[[ITER:[a-zA-Z0-9]+]]
// CHECK-MASK-DAG:       %[[DEST_SZ1:.+]] = affine.min #[[$MAP1]]
// CHECK-MASK-DAG:       %[[DEST_SZ2:.+]] = affine.min #[[$MAP2]]
// CHECK-MASK-DAG:       %[[SRC_SZ0:.+]] = affine.min #[[$MAP4]]
// CHECK-MASK-DAG:       %[[SRC_SZ2:.+]] = affine.min #[[$MAP3]]
// CHECK-MASK-DAG:       %[[ITER_SLICE:.+]] = tensor.extract_slice %[[GENERIC_EMPTY]]
// CHECK-MASK-DAG:       %[[SRC_SLICE:.+]] = tensor.extract_slice %[[SRC]][{{.+}}] [%[[SRC_SZ0]], 2, %[[SRC_SZ2]]]
// CHECK-MASK-DAG:       %[[READ_MASK:.+]] = vector.create_mask %[[SRC_SZ0]], %[[C2]], %[[SRC_SZ2]] : vector<64x2x12xi1>
// CHECK-MASK:           %[[GENERIC_READ:.+]] = vector.transfer_read %[[SRC_SLICE]]{{.+}} %[[READ_MASK]]
// CHECK-MASK-DAG:       %[[WRITE_MASK:.+]] = vector.create_mask %[[C2]], %[[SRC_SZ2]], %[[SRC_SZ0]] : vector<2x12x64xi1>
// CHECK-MASK:           %[[TRUNC:.+]] = arith.truncf %[[GENERIC_READ]]
// CHECK-MASK:           %[[TRANSP:.+]] = vector.transpose %[[TRUNC]], [1, 2, 0]
// CHECK-MASK:           %[[GENERIC_WRITE:.+]] = vector.transfer_write %[[TRANSP]], %[[ITER_SLICE]]{{.+}}, %[[WRITE_MASK]]
// CHECK-MASK:           %[[D1:.+]] = tensor.dim %[[GENERIC_WRITE]], %[[C1]]
// CHECK-MASK:           %[[D2:.+]] = tensor.dim %[[GENERIC_WRITE]], %[[C2]]
// CHECK-MASK:           %[[PACK_READ_MASK:.+]] = vector.create_mask %[[C2]], %[[D1]], %[[D2]] : vector<2x12x64xi1>
// CHECK-MASK:           %[[PACK_SRC:.+]] = vector.transfer_read %[[GENERIC_WRITE]]{{.+}}, %[[PACK_READ_MASK]]
// CHECK-MASK:           %[[SHAPE_CAST:.+]] = vector.shape_cast %[[PACK_SRC]] : vector<2x12x64xbf16> to vector<2x6x2x4x16xbf16>
// CHECK-MASK:           %[[PACK_TRANSP:.+]] = vector.transpose %[[SHAPE_CAST]], [0, 3, 1, 4, 2]
// CHECK-MASK:           %[[EMPTY:.+]] = tensor.empty(%[[DEST_SZ1]], %[[DEST_SZ2]]) : tensor<2x?x?x16x2xbf16>
// CHECK-MASK:           %[[PACK_WRITE_MASK:.+]] = vector.create_mask %[[C2]], %[[DEST_SZ1]], %[[DEST_SZ2]], %[[C16]], %[[C2]] : vector<2x4x6x16x2xi1>
// CHECK-MASK:           vector.transfer_write %[[PACK_TRANSP]], %[[EMPTY]]{{.+}}, %[[PACK_WRITE_MASK]]

// -----

#map = affine_map<(d0)[s0] -> (16, -d0 + s0)>
#map1 = affine_map<(d0)[s0] -> (32, -d0 + s0)>
#map2 = affine_map<(d0) -> (d0 floordiv 16)>
#map3 = affine_map<(d0) -> (d0 ceildiv 16)>
func.func @single_dynamic_unpack_infer_vector_size(%arg0: tensor<?x?x16x16xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg1, %c0 : tensor<?x?xf32>
  %c1 = arith.constant 1 : index
  %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %c0_1 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %0 = scf.for %arg2 = %c0_1 to %dim step %c16 iter_args(%arg3 = %arg1) -> (tensor<?x?xf32>) {
    %c0_2 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %1 = scf.for %arg4 = %c0_2 to %dim_0 step %c32 iter_args(%arg5 = %arg3) -> (tensor<?x?xf32>) {
      %2 = affine.min #map(%arg2)[%dim]
      %3 = affine.min #map1(%arg4)[%dim_0]
      %4 = affine.apply #map2(%arg2)
      %5 = affine.apply #map2(%arg4)
      %6 = affine.apply #map3(%3)
      %extracted_slice = tensor.extract_slice %arg0[%4, %5, 0, 0] [1, %6, 16, 16] [1, 1, 1, 1] : tensor<?x?x16x16xf32> to tensor<1x?x16x16xf32>
      %extracted_slice_3 = tensor.extract_slice %arg5[%arg2, %arg4] [%2, %3] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %unpack = tensor.unpack %extracted_slice outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %extracted_slice_3 : tensor<1x?x16x16xf32> -> tensor<?x?xf32>
      %inserted_slice = tensor.insert_slice %unpack into %arg5[%arg2, %arg4] [%2, %3] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
      scf.yield %inserted_slice : tensor<?x?xf32>
    }
    scf.yield %1 : tensor<?x?xf32>
  }
  return %0 : tensor<?x?xf32>
}
// CHECK-MASK: #[[$MAP0:.+]] = affine_map<(d0)[s0] -> (16, -d0 + s0)>
// CHECK-MASK: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (32, -d0 + s0)>
// CHECK-MASK: #[[$MAP2:.+]] = affine_map<(d0) -> (d0 floordiv 16)>
// CHECK-MASK: #[[$MAP3:.+]] = affine_map<(d0) -> (d0 ceildiv 16)>
// CHECK-MASK-LABEL: func.func @single_dynamic_unpack_infer_vector_size
// CHECK-MASK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-MASK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-MASK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-MASK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK-MASK:         scf.for
// CHECK-MASK:         scf.for
// CHECK-MASK-DAG:       %[[DEST_SZ0:.+]] = affine.min #[[$MAP0]]
// CHECK-MASK-DAG:       %[[DEST_SZ1:.+]] = affine.min #[[$MAP1]]
// CHECK-MASK-DAG:       %[[SRC_SZ1:.+]] = affine.apply #[[$MAP3]]
// CHECK-MASK:           %[[SRC_SLICE:.+]] = tensor.extract_slice %[[SRC]]
// CHECK-MASK:           %[[READ_MASK:.+]] = vector.create_mask %[[C1]], %[[SRC_SZ1]], %[[C16]], %[[C16]] : vector<1x2x16x16xi1>
// CHECK-MASK:           %[[READ:.+]] = vector.transfer_read %[[SRC_SLICE]]{{.+}}, %[[READ_MASK]]
// CHECK-MASK:           %[[TRANSP:.+]] = vector.transpose %[[READ]], [0, 2, 1, 3]
// CHECK-MASK:           %[[SHAPE_CAST:.+]] = vector.shape_cast %[[TRANSP]] : vector<1x16x2x16xf32> to vector<16x32xf32>
// CHECK-MASK:           %[[EMPTY:.+]] = tensor.empty(%[[DEST_SZ0]], %[[DEST_SZ1]]) : tensor<?x?xf32>
// CHECK-MASK:           %[[WRITE_MASK:.+]] = vector.create_mask %[[DEST_SZ0]], %[[DEST_SZ1]] : vector<16x32xi1>
// CHECK-MASK:           vector.transfer_write %[[SHAPE_CAST]], {{.+}}, %[[WRITE_MASK]]

// -----

#map = affine_map<(d0)[s0] -> (-d0 + s0, 16)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 32)>
#map2 = affine_map<(d0) -> (d0 floordiv 16)>
#map3 = affine_map<(d0) -> (d0 ceildiv 16)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
func.func @generic_unpack_infer_vector_size(%arg0: tensor<?x?x16x16xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c32 = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg1, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %0 = scf.for %arg3 = %c0 to %dim step %c16 iter_args(%arg4 = %arg2) -> (tensor<?x?xf32>) {
    %1 = scf.for %arg5 = %c0 to %dim_0 step %c32 iter_args(%arg6 = %arg4) -> (tensor<?x?xf32>) {
      %2 = affine.min #map(%arg3)[%dim]
      %3 = affine.min #map1(%arg5)[%dim_0]
      %4 = affine.apply #map2(%arg3)
      %5 = affine.apply #map2(%arg5)
      %6 = affine.apply #map3(%3)
      %extracted_slice = tensor.extract_slice %arg0[%4, %5, 0, 0] [1, %6, 16, 16] [1, 1, 1, 1] : tensor<?x?x16x16xf32> to tensor<1x?x16x16xf32>
      %extracted_slice_1 = tensor.extract_slice %arg1[%arg3, %arg5] [%2, %3] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %unpack = tensor.unpack %extracted_slice outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %extracted_slice_1 : tensor<1x?x16x16xf32> -> tensor<?x?xf32>
      %extracted_slice_2 = tensor.extract_slice %arg6[%arg3, %arg5] [%2, %3] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %7 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%unpack : tensor<?x?xf32>) outs(%extracted_slice_2 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %8 = math.exp %in : f32
        linalg.yield %8 : f32
      } -> tensor<?x?xf32>
      %inserted_slice = tensor.insert_slice %7 into %arg6[%arg3, %arg5] [%2, %3] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
      scf.yield %inserted_slice : tensor<?x?xf32>
    }
    scf.yield %1 : tensor<?x?xf32>
  }
  return %0 : tensor<?x?xf32>
}
// CHECK-MASK: #[[$MAP0:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 16)>
// CHECK-MASK: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 32)>
// CHECK-MASK: #[[$MAP2:.+]] = affine_map<(d0) -> (d0 floordiv 16)>
// CHECK-MASK: #[[$MAP3:.+]] = affine_map<(d0) -> (d0 ceildiv 16)>
// CHECK-MASK-LABEL: func.func @generic_unpack_infer_vector_size
// CHECK-MASK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-MASK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-MASK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-MASK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK-MASK:         scf.for
// CHECK-MASK:         scf.for
// CHECK-MASK-DAG:       %[[DEST_SZ0:.+]] = affine.min #[[$MAP0]]
// CHECK-MASK-DAG:       %[[DEST_SZ1:.+]] = affine.min #[[$MAP1]]
// CHECK-MASK-DAG:       %[[SRC_SZ1:.+]] = affine.apply #[[$MAP3]]
// CHECK-MASK:           %[[SRC_SLICE:.+]] = tensor.extract_slice %[[SRC]]
// CHECK-MASK:           %[[READ_MASK:.+]] = vector.create_mask %[[C1]], %[[SRC_SZ1]], %[[C16]], %[[C16]] : vector<1x2x16x16xi1>
// CHECK-MASK:           %[[READ:.+]] = vector.transfer_read %[[SRC_SLICE]]{{.+}}, %[[READ_MASK]]
// CHECK-MASK:           %[[TRANSP:.+]] = vector.transpose %[[READ]], [0, 2, 1, 3]
// CHECK-MASK:           %[[SHAPE_CAST:.+]] = vector.shape_cast %[[TRANSP]] : vector<1x16x2x16xf32> to vector<16x32xf32>
// CHECK-MASK:           %[[EMPTY:.+]] = tensor.empty(%[[DEST_SZ0]], %[[DEST_SZ1]]) : tensor<?x?xf32>
// CHECK-MASK:           %[[WRITE_MASK:.+]] = vector.create_mask %[[DEST_SZ0]], %[[DEST_SZ1]] : vector<16x32xi1>
// CHECK-MASK:           %[[UNPACK_WRITE:.+]] = vector.transfer_write %[[SHAPE_CAST]], {{.+}}, %[[WRITE_MASK]]
// CHECK-MASK:           %[[D0:.+]] = tensor.dim %[[UNPACK_WRITE]], %[[C0]]
// CHECK-MASK:           %[[D1:.+]] = tensor.dim %[[UNPACK_WRITE]], %[[C1]]
// CHECK-MASK:           %[[GENERIC_MASK:.+]] = vector.create_mask %[[D0]], %[[D1]] : vector<16x32xi1>
// CHECK-MASK:           %[[GENERIC_SRC:.+]] = vector.transfer_read %[[UNPACK_WRITE]]{{.+}}, %[[GENERIC_MASK]]
// CHECK-MASK:           %[[EXP:.+]] = math.exp %[[GENERIC_SRC]]
// CHECK-MASK:           vector.transfer_write %[[EXP]]{{.+}}, %[[GENERIC_MASK]]

// -----

#aarch64_sve = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve", target_triple = "aarch64-none-elf"}>
#map = affine_map<()[s0] -> (-(176 mod s0) + 176)>

func.func @dynamic_fill_with_scalable_tiling_infer_vector_size(%arg0: tensor<1x67x120x176xf32>) -> tensor<1x67x120x176xf32>
  attributes {hal.executable.target = #aarch64_sve}
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c67 = arith.constant 67 : index
  %c120 = arith.constant 120 : index
  %cst = arith.constant 0.000000e+00 : f32
  %vscale = vector.vscale
  %c4_vscale = arith.muli %vscale, %c4 : index
  %0 = scf.for %arg1 = %c0 to %c67 step %c1 iter_args(%arg2 = %arg0) -> (tensor<1x67x120x176xf32>) {
    %1 = scf.for %arg3 = %c0 to %c120 step %c4 iter_args(%arg4 = %arg2) -> (tensor<1x67x120x176xf32>) {
      %2 = affine.apply #map()[%c4_vscale]
      %3 = scf.for %arg5 = %c0 to %2 step %c4_vscale iter_args(%arg6 = %arg4) -> (tensor<1x67x120x176xf32>) {
        %extracted_slice = tensor.extract_slice %arg6[0, %arg1, %arg3, %arg5] [1, 1, 4, %c4_vscale] [1, 1, 1, 1] : tensor<1x67x120x176xf32> to tensor<1x1x4x?xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%extracted_slice : tensor<1x1x4x?xf32>) -> tensor<1x1x4x?xf32>
        %inserted_slice = tensor.insert_slice %4 into %arg6[0, %arg1, %arg3, %arg5] [1, 1, 4, %c4_vscale] [1, 1, 1, 1] : tensor<1x1x4x?xf32> into tensor<1x67x120x176xf32>
        scf.yield %inserted_slice : tensor<1x67x120x176xf32>
      }
      scf.yield %3 : tensor<1x67x120x176xf32>
    }
    scf.yield %1 : tensor<1x67x120x176xf32>
  }
  return %0 : tensor<1x67x120x176xf32>
}

// CHECK-MASK-LABEL: func.func @dynamic_fill_with_scalable_tiling_infer_vector_size
// CHECK-MASK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<1x1x4x[4]xf32>
// CHECK-MASK: scf.for
// CHECK-MASK:   scf.for
// CHECK-MASK:     scf.for
// CHECK-MASK:       vector.transfer_write %[[CST]], {{.*}} {in_bounds = [true, true, true, true]} : vector<1x1x4x[4]xf32>, tensor<1x1x4x?xf32>

// -----

#aarch64_sve = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve", target_triple = "aarch64-none-elf"}>

func.func @dynamic_fill_with_scalable_tiling_infer_remainder_vector_size(%arg0: tensor<1x67x120x100xf32>) -> tensor<1x67x120x100xf32>
  attributes {hal.executable.target = #aarch64_sve}
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c100 = arith.constant 100 : index
  %c67 = arith.constant 67 : index
  %c120 = arith.constant 120 : index
  %cst = arith.constant 0.000000e+00 : f32
  %vscale = vector.vscale
  %c4_vscale = arith.muli %vscale, %c4 : index
  %0 = scf.for %arg1 = %c0 to %c67 step %c1 iter_args(%arg2 = %arg0) -> (tensor<1x67x120x100xf32>) {
    %1 = scf.for %arg3 = %c0 to %c120 step %c4 iter_args(%arg4 = %arg2) -> (tensor<1x67x120x100xf32>) {
      %rem_start = affine.apply affine_map<()[s0] -> (-(100 mod s0) + 100)>()[%c4_vscale]
      %3 = scf.for %arg5 = %rem_start to %c100 step %c4_vscale iter_args(%arg6 = %arg4) -> (tensor<1x67x120x100xf32>) {
        %rem_elts = affine.apply affine_map<(d0) -> (-d0 + 100)>(%arg5)
        %extracted_slice = tensor.extract_slice %arg6[0, %arg1, %arg3, %arg5] [1, 1, 4, %rem_elts] [1, 1, 1, 1] : tensor<1x67x120x100xf32> to tensor<1x1x4x?xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%extracted_slice : tensor<1x1x4x?xf32>) -> tensor<1x1x4x?xf32>
        %inserted_slice = tensor.insert_slice %4 into %arg6[0, %arg1, %arg3, %arg5] [1, 1, 4, %rem_elts] [1, 1, 1, 1] : tensor<1x1x4x?xf32> into tensor<1x67x120x100xf32>
        scf.yield %inserted_slice : tensor<1x67x120x100xf32>
      }
      scf.yield %3 : tensor<1x67x120x100xf32>
    }
    scf.yield %1 : tensor<1x67x120x100xf32>
  }
  return %0 : tensor<1x67x120x100xf32>
}

// CHECK-MASK-LABEL: func.func @dynamic_fill_with_scalable_tiling_infer_remainder_vector_size
// CHECK-MASK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<1x1x4x[4]xf32>
// CHECK-MASK: scf.for
// CHECK-MASK:   scf.for
// CHECK-MASK:     scf.for
// CHECK-MASK:       vector.transfer_write %[[CST]], {{.*}} {in_bounds = [true, true, true, true]} : vector<1x1x4x[4]xf32>, tensor<1x1x4x?xf32>

// -----

#aarch64_sve = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve", target_triple = "aarch64-none-elf"}>
#config = #iree_codegen.lowering_config<tile_sizes = [[0, 0, 0, 0], [1, 4, [4], 0], [0, 0, 0, 3], [0, 0, 0, 0]]>
#map = affine_map<()[s0] -> (-(96 mod s0) + 96)>
#map1 = affine_map<(d0) -> (d0 * 2)>

func.func @depthwise_conv_fold_away_masking(%arg0: tensor<1x68x120x96xf32>, %arg1: tensor<1x137x241x96xf32>, %arg2: tensor<3x3x96xf32>) -> tensor<1x68x120x96xf32>
  attributes {hal.executable.target = #aarch64_sve}
{
  %c3 = arith.constant 3 : index
  %c120 = arith.constant 120 : index
  %c68 = arith.constant 68 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %vscale = vector.vscale
  %c4_vscale = arith.muli %vscale, %c4 : index
  %0 = scf.for %arg3 = %c0 to %c68 step %c1 iter_args(%arg4 = %arg0) -> (tensor<1x68x120x96xf32>) {
    %1 = scf.for %arg5 = %c0 to %c120 step %c4 iter_args(%arg6 = %arg4) -> (tensor<1x68x120x96xf32>) {
      %2 = affine.apply #map()[%c4_vscale]
      %3 = scf.for %arg7 = %c0 to %2 step %c4_vscale iter_args(%arg8 = %arg6) -> (tensor<1x68x120x96xf32>) {
        %4 = affine.apply #map1(%arg3)
        %5 = affine.apply #map1(%arg5)
        %extracted_slice = tensor.extract_slice %arg1[0, %4, %5, %arg7] [1, 3, 9, %c4_vscale] [1, 1, 1, 1] : tensor<1x137x241x96xf32> to tensor<1x3x9x?xf32>
        %extracted_slice_0 = tensor.extract_slice %arg2[0, 0, %arg7] [3, 3, %c4_vscale] [1, 1, 1] : tensor<3x3x96xf32> to tensor<3x3x?xf32>
        %extracted_slice_1 = tensor.extract_slice %arg8[0, %arg3, %arg5, %arg7] [1, 1, 4, %c4_vscale] [1, 1, 1, 1] : tensor<1x68x120x96xf32> to tensor<1x1x4x?xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%extracted_slice_1 : tensor<1x1x4x?xf32>) -> tensor<1x1x4x?xf32>
        %7 = scf.for %arg9 = %c0 to %c3 step %c1 iter_args(%arg10 = %6) -> (tensor<1x1x4x?xf32>) {
          %extracted_slice_2 = tensor.extract_slice %extracted_slice[0, %arg9, 0, 0] [1, 1, 9, %c4_vscale] [1, 1, 1, 1] : tensor<1x3x9x?xf32> to tensor<1x1x9x?xf32>
          %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[%arg9, 0, 0] [1, 3, %c4_vscale] [1, 1, 1] : tensor<3x3x?xf32> to tensor<1x3x?xf32>
          %extracted_slice_4 = tensor.extract_slice %arg10[0, 0, 0, 0] [1, 1, 4, %c4_vscale] [1, 1, 1, 1] : tensor<1x1x4x?xf32> to tensor<1x1x4x?xf32>
          %extracted_slice_5 = tensor.extract_slice %extracted_slice_2[0, 0, 0, 0] [1, 1, 9, %c4_vscale] [1, 1, 1, 1] : tensor<1x1x9x?xf32> to tensor<1x9x?xf32>
          %extracted_slice_6 = tensor.extract_slice %extracted_slice_3[0, 0, 0] [1, 3, %c4_vscale] [1, 1, 1] : tensor<1x3x?xf32> to tensor<3x?xf32>
          %extracted_slice_7 = tensor.extract_slice %extracted_slice_4[0, 0, 0, 0] [1, 1, 4, %c4_vscale] [1, 1, 1, 1] : tensor<1x1x4x?xf32> to tensor<1x4x?xf32>
          %8 = linalg.depthwise_conv_1d_nwc_wc {dilations = dense<1> : vector<1xi64>, lowering_config = #config, strides = dense<2> : vector<1xi64>} ins(%extracted_slice_5, %extracted_slice_6 : tensor<1x9x?xf32>, tensor<3x?xf32>) outs(%extracted_slice_7 : tensor<1x4x?xf32>) -> tensor<1x4x?xf32>
          %inserted_slice_8 = tensor.insert_slice %8 into %extracted_slice_4[0, 0, 0, 0] [1, 1, 4, %c4_vscale] [1, 1, 1, 1] : tensor<1x4x?xf32> into tensor<1x1x4x?xf32>
          %inserted_slice_9 = tensor.insert_slice %inserted_slice_8 into %arg10[0, 0, 0, 0] [1, 1, 4, %c4_vscale] [1, 1, 1, 1] : tensor<1x1x4x?xf32> into tensor<1x1x4x?xf32>
          scf.yield %inserted_slice_9 : tensor<1x1x4x?xf32>
        }
        %inserted_slice = tensor.insert_slice %7 into %arg8[0, %arg3, %arg5, %arg7] [1, 1, 4, %c4_vscale] [1, 1, 1, 1] : tensor<1x1x4x?xf32> into tensor<1x68x120x96xf32>
        scf.yield %inserted_slice : tensor<1x68x120x96xf32>
      }
      scf.yield %3 : tensor<1x68x120x96xf32>
    }
    scf.yield %1 : tensor<1x68x120x96xf32>
  }
  return %0 : tensor<1x68x120x96xf32>
}

/// This checks that the masks (introduced by the vectorizer) are eliminated by
/// the end of the iree-codegen-generic-vectorization pass.

// CHECK-MASK-LABEL: func.func @depthwise_conv_fold_away_masking
// CHECK-MASK-NOT: vector.create_mask
// CHECK-MASK-NOT: vector.constant_mask
// CHECK-MASK:     vector.fma
// CHECK-MASK-NOT: vector.create_mask
// CHECK-MASK-NOT: vector.constant_mask
