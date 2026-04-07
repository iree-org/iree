// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-generic-vectorization{enable-vector-masking=true use-configured-vector-sizes=false}))" --split-input-file %s | FileCheck %s

// Tests for masked vectorization with vector sizes inferred from the IR.
// No lowering_config is attached to the ops; vector sizes are derived from
// tensor shapes and loop structure by the pass's vector size inference logic.

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
      %pack = linalg.pack %extracted_slice padding_value(%c0_i8 : i8) outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 2] into %extracted_slice_0 : tensor<?x?xi8> -> tensor<?x?x16x2xi8>
      %inserted_slice = tensor.insert_slice %pack into %arg5[%arg2, %arg4, 0, 0] [%2, %3, 16, 2] [1, 1, 1, 1] : tensor<?x?x16x2xi8> into tensor<13x51x16x2xi8>
      scf.yield %inserted_slice : tensor<13x51x16x2xi8>
    }
    scf.yield %1 : tensor<13x51x16x2xi8>
  }
  return %0 : tensor<13x51x16x2xi8>
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (-d0 + 13, 2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (-d0 + 51, 4)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK: #[[$MAP3:.+]] = affine_map<(d0, d1) -> (d1 * -2 + 101, d0 * 2)>
// CHECK: #[[$MAP4:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK: #[[$MAP5:.+]] = affine_map<(d0, d1) -> (d1 * -16 + 201, d0 * 16)>
// CHECK-LABEL: func.func @single_static_pack_infer_vector_size
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C0_I8:.+]] = arith.constant 0 : i8
// CHECK:         scf.for
// CHECK:           scf.for {{.*}} iter_args(%[[ITER:.+]] = %{{.*}})
// CHECK-DAG:         %[[WRITE_SZ0:.+]] = affine.min #[[$MAP0]]
// CHECK-DAG:         %[[WRITE_SZ1:.+]] = affine.min #[[$MAP1]]
// CHECK-DAG:         %[[READ_SZ0:.+]] = affine.min #[[$MAP3]]
// CHECK-DAG:         %[[READ_SZ1:.+]] = affine.min #[[$MAP5]]
// CHECK-DAG:         %[[SLICE:.+]] = tensor.extract_slice %[[SRC]][{{.+}}] [%[[READ_SZ0]], %[[READ_SZ1]]]
// CHECK-DAG:         %[[W_SLICE:.+]] = tensor.extract_slice %[[ITER]][{{.+}}] [%[[WRITE_SZ0]], %[[WRITE_SZ1]], 16, 2]
// CHECK:             %[[READ_MASK:.+]] = vector.create_mask %[[READ_SZ0]], %[[READ_SZ1]] : vector<8x32xi1>
// CHECK:             %[[READ:.+]] = vector.transfer_read %[[SLICE]][%{{.+}}], %[[C0_I8]], %[[READ_MASK]]
// CHECK:             %[[CAST:.+]] = vector.shape_cast %[[READ]] : vector<8x32xi8> to vector<4x2x2x16xi8>
// CHECK:             %[[TRANSP:.+]] = vector.transpose %[[CAST]], [2, 0, 3, 1]
// CHECK:             %[[WRITE_MASK:.+]] = vector.create_mask %[[WRITE_SZ0]], %[[WRITE_SZ1]], {{.+}} : vector<2x4x16x2xi1>
// CHECK:             vector.transfer_write %[[TRANSP]], %[[W_SLICE]][{{.+}}, %[[WRITE_MASK]]

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
      %pack = linalg.pack %extracted_slice padding_value(%c0_i8 : i8) outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 2] into %extracted_slice_3 : tensor<?x?xi8> -> tensor<?x?x16x2xi8>
      %inserted_slice = tensor.insert_slice %pack into %arg5[%arg2, %arg4, 0, 0] [%2, %3, 16, 2] [1, 1, 1, 1] : tensor<?x?x16x2xi8> into tensor<?x?x16x2xi8>
      scf.yield %inserted_slice : tensor<?x?x16x2xi8>
    }
    scf.yield %1 : tensor<?x?x16x2xi8>
  }
  return %0 : tensor<?x?x16x2xi8>
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 4)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK: #[[$MAP3:.+]] = affine_map<(d0, d1)[s0] -> (d1 * -2 + s0, d0 * 2)>
// CHECK: #[[$MAP4:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK: #[[$MAP5:.+]] = affine_map<(d0, d1)[s0] -> (d1 * -16 + s0, d0 * 16)>
// CHECK-LABEL: func.func @single_dynamic_pack_infer_vector_size
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C0_I8:.+]] = arith.constant 0 : i8
// CHECK:         scf.for
// CHECK:           scf.for {{.*}} iter_args(%[[ITER:.+]] = {{.*}})
// CHECK-DAG:         %[[WRITE_SZ0:.+]] = affine.min #[[$MAP0]]
// CHECK-DAG:         %[[WRITE_SZ1:.+]] = affine.min #[[$MAP1]]
// CHECK-DAG:         %[[READ_SZ0:.+]] = affine.min #[[$MAP3]]
// CHECK-DAG:         %[[READ_SZ1:.+]] = affine.min #[[$MAP5]]
// CHECK-DAG:         %[[SLICE:.+]] = tensor.extract_slice %[[SRC]][{{.+}}] [%[[READ_SZ0]], %[[READ_SZ1]]]
// CHECK-DAG:         %[[W_SLICE:.+]] = tensor.extract_slice %[[ITER]][{{.+}}] [%[[WRITE_SZ0]], %[[WRITE_SZ1]], 16, 2]
// CHECK:             %[[READ_MASK:.+]] = vector.create_mask %[[READ_SZ0]], %[[READ_SZ1]] : vector<8x32xi1>
// CHECK:             %[[READ:.+]] = vector.transfer_read %[[SLICE]][%{{.+}}], %[[C0_I8]], %[[READ_MASK]]
// CHECK:             %[[CAST:.+]] = vector.shape_cast %[[READ]] : vector<8x32xi8> to vector<4x2x2x16xi8>
// CHECK:             %[[TRANSP:.+]] = vector.transpose %[[CAST]], [2, 0, 3, 1]
// CHECK:             %[[WRITE_MASK:.+]] = vector.create_mask %[[WRITE_SZ0]], %[[WRITE_SZ1]], {{.+}} : vector<2x4x16x2xi1>
// CHECK:             vector.transfer_write %[[TRANSP]], %[[W_SLICE]][{{.+}}, %[[WRITE_MASK]]

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
        %pack = linalg.pack %12 padding_value(%cst : bf16) outer_dims_perm = [0, 2, 1] inner_dims_pos = [2, 1] inner_tiles = [16, 2] into %extracted_slice_1 : tensor<2x?x?xbf16> -> tensor<2x?x?x16x2xbf16>
        %inserted_slice = tensor.insert_slice %pack into %arg6[%arg1, %arg3, %arg5, 0, 0] [2, %6, %7, 16, 2] [1, 1, 1, 1, 1] : tensor<2x?x?x16x2xbf16> into tensor<32x?x64x16x2xbf16>
        scf.yield %inserted_slice : tensor<32x?x64x16x2xbf16>
      }
      scf.yield %5 : tensor<32x?x64x16x2xbf16>
    }
    scf.yield %4 : tensor<32x?x64x16x2xbf16>
  }
  return %3 : tensor<32x?x64x16x2xbf16>
}
// CHECK: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (4, -d0 + s0 ceildiv 16)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0) -> (-d0 + 64, 6)>
// CHECK: #[[$MAP3:.+]] = affine_map<(d0, d1) -> (d1 * -2 + 128, d0 * 2)>
// CHECK: #[[$MAP4:.+]] = affine_map<(d0, d1)[s0] -> (d1 * -16 + s0, d0 * 16)>
// CHECK: #[[$MAP5:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK: #[[$MAP6:.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL: func.func @generic_pack_infer_vector_size
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0_BF16:.+]] = arith.constant 0.000000e+00 : bf16
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK:         %[[D0:.+]] = tensor.dim %[[SRC]], %[[C0]] : tensor<?x32x128xf32>
// CHECK:         %[[GENERIC_EMPTY:.+]] = tensor.empty(%[[D0]]) : tensor<32x128x?xbf16>
// CHECK:         scf.for %[[I0:.+]] = %c0
// CHECK:         scf.for %[[I1:.+]] = %c0
// CHECK:         scf.for %[[I2:.+]] = %c0
// CHECK-SAME:      iter_args(%[[ITER:[a-zA-Z0-9]+]]
// CHECK-DAG:       %[[DEST_SZ1:.+]] = affine.min #[[$MAP1]]
// CHECK-DAG:       %[[DEST_SZ2:.+]] = affine.min #[[$MAP2]]
// CHECK-DAG:       %[[SRC_SZ0:.+]] = affine.min #[[$MAP4]]
// CHECK-DAG:       %[[SRC_SZ2:.+]] = affine.min #[[$MAP3]]
// CHECK-DAG:       %[[EMPTY_SLICE:.+]] = tensor.extract_slice %[[GENERIC_EMPTY]]
// CHECK-DAG:       %[[SRC_SLICE:.+]] = tensor.extract_slice %[[SRC]][{{.+}}] [%[[SRC_SZ0]], 2, %[[SRC_SZ2]]]
// CHECK-DAG:       %[[READ_MASK:.+]] = vector.create_mask %[[SRC_SZ0]], %[[C2]], %[[SRC_SZ2]] : vector<64x2x12xi1>
// CHECK:           %[[GENERIC_READ:.+]] = vector.transfer_read %[[SRC_SLICE]]{{.+}} %[[READ_MASK]]
// CHECK-DAG:       %[[WRITE_MASK:.+]] = vector.create_mask %[[C2]], %[[SRC_SZ2]], %[[SRC_SZ0]] : vector<2x12x64xi1>
// CHECK:           %[[TRUNC:.+]] = arith.truncf %[[GENERIC_READ]]
// CHECK:           %[[TRANSP:.+]] = vector.transpose %[[TRUNC]], [1, 2, 0]
// CHECK:           %[[GENERIC_WRITE:.+]] = vector.transfer_write %[[TRANSP]], %[[EMPTY_SLICE]]{{.+}}, %[[WRITE_MASK]]
// CHECK:           %[[W_SLICE:.+]] = tensor.extract_slice %[[ITER]][%[[I0]], %[[I1]], %[[I2]], 0, 0] [2, %[[DEST_SZ1]], %[[DEST_SZ2]], 16, 2]
// CHECK:           %[[D1:.+]] = tensor.dim %[[GENERIC_WRITE]], %[[C1]]
// CHECK:           %[[D2:.+]] = tensor.dim %[[GENERIC_WRITE]], %[[C2]]
// CHECK:           %[[PACK_READ_MASK:.+]] = vector.create_mask %[[C2]], %[[D1]], %[[D2]] : vector<2x12x64xi1>
// CHECK:           %[[PACK_SRC:.+]] = vector.transfer_read %[[GENERIC_WRITE]]{{.+}}, %[[PACK_READ_MASK]]
// CHECK:           %[[SHAPE_CAST:.+]] = vector.shape_cast %[[PACK_SRC]] : vector<2x12x64xbf16> to vector<2x6x2x4x16xbf16>
// CHECK:           %[[PACK_TRANSP:.+]] = vector.transpose %[[SHAPE_CAST]], [0, 3, 1, 4, 2]
// CHECK:           %[[PACK_WRITE_MASK:.+]] = vector.create_mask %[[C2]], %[[DEST_SZ1]], %[[DEST_SZ2]], %[[C16]], %[[C2]] : vector<2x4x6x16x2xi1>
// CHECK:           vector.transfer_write %[[PACK_TRANSP]], %[[W_SLICE]]{{.+}}, %[[PACK_WRITE_MASK]]

// -----

#map = affine_map<(d0)[s0] -> (16, -d0 + s0)>
#map1 = affine_map<(d0)[s0] -> (32, -d0 + s0)>
#map2 = affine_map<(d0) -> (d0 floordiv 16)>
#map3 = affine_map<(d0) -> (d0 ceildiv 16)>
func.func @single_dynamic_unpack_infer_vector_size(%arg0: tensor<?x?x16x16xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %dim = tensor.dim %arg1, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %0 = scf.for %arg2 = %c0 to %dim step %c16 iter_args(%arg3 = %arg1) -> (tensor<?x?xf32>) {
    %1 = scf.for %arg4 = %c0 to %dim_0 step %c32 iter_args(%arg5 = %arg3) -> (tensor<?x?xf32>) {
      %2 = affine.min #map(%arg2)[%dim]
      %3 = affine.min #map1(%arg4)[%dim_0]
      %4 = affine.apply #map2(%arg2)
      %5 = affine.apply #map2(%arg4)
      %6 = affine.apply #map3(%3)
      %extracted_slice = tensor.extract_slice %arg0[%4, %5, 0, 0] [1, %6, 16, 16] [1, 1, 1, 1] : tensor<?x?x16x16xf32> to tensor<1x?x16x16xf32>
      %extracted_slice_3 = tensor.extract_slice %arg5[%arg2, %arg4] [%2, %3] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %unpack = linalg.unpack %extracted_slice outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %extracted_slice_3 : tensor<1x?x16x16xf32> -> tensor<?x?xf32>
      %inserted_slice = tensor.insert_slice %unpack into %arg5[%arg2, %arg4] [%2, %3] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
      scf.yield %inserted_slice : tensor<?x?xf32>
    }
    scf.yield %1 : tensor<?x?xf32>
  }
  return %0 : tensor<?x?xf32>
}
// CHECK: #[[$MAP0:.+]] = affine_map<(d0)[s0] -> (16, -d0 + s0)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (32, -d0 + s0)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0) -> (d0 floordiv 16)>
// CHECK: #[[$MAP3:.+]] = affine_map<(d0) -> (d0 ceildiv 16)>
// CHECK-LABEL: func.func @single_dynamic_unpack_infer_vector_size
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK:         scf.for
// CHECK:         scf.for
// CHECK-DAG:       %[[DEST_SZ0:.+]] = affine.min #[[$MAP0]]
// CHECK-DAG:       %[[DEST_SZ1:.+]] = affine.min #[[$MAP1]]
// CHECK-DAG:       %[[SRC_SZ1:.+]] = affine.apply #[[$MAP3]]
// CHECK:           %[[SRC_SLICE:.+]] = tensor.extract_slice %[[SRC]]
// CHECK:           %[[READ_MASK:.+]] = vector.create_mask %[[C1]], %[[SRC_SZ1]], %[[C16]], %[[C16]] : vector<1x2x16x16xi1>
// CHECK:           %[[READ:.+]] = vector.transfer_read %[[SRC_SLICE]]{{.+}}, %[[READ_MASK]]
// CHECK:           %[[TRANSP:.+]] = vector.transpose %[[READ]], [0, 2, 1, 3]
// CHECK:           %[[SHAPE_CAST:.+]] = vector.shape_cast %[[TRANSP]] : vector<1x16x2x16xf32> to vector<16x32xf32>
// CHECK:           %[[WRITE_MASK:.+]] = vector.create_mask %[[DEST_SZ0]], %[[DEST_SZ1]] : vector<16x32xi1>
// CHECK:           vector.transfer_write %[[SHAPE_CAST]], {{.+}}, %[[WRITE_MASK]]

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
      %unpack = linalg.unpack %extracted_slice outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %extracted_slice_1 : tensor<1x?x16x16xf32> -> tensor<?x?xf32>
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
// CHECK: #[[$MAP0:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 16)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 32)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0) -> (d0 floordiv 16)>
// CHECK: #[[$MAP3:.+]] = affine_map<(d0) -> (d0 ceildiv 16)>
// CHECK-LABEL: func.func @generic_unpack_infer_vector_size
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK:         scf.for
// CHECK:         scf.for
// CHECK-DAG:       %[[DEST_SZ0:.+]] = affine.min #[[$MAP0]]
// CHECK-DAG:       %[[DEST_SZ1:.+]] = affine.min #[[$MAP1]]
// CHECK-DAG:       %[[SRC_SZ1:.+]] = affine.apply #[[$MAP3]]
// CHECK:           %[[SRC_SLICE:.+]] = tensor.extract_slice %[[SRC]]
// CHECK:           %[[READ_MASK:.+]] = vector.create_mask %[[C1]], %[[SRC_SZ1]], %[[C16]], %[[C16]] : vector<1x2x16x16xi1>
// CHECK:           %[[READ:.+]] = vector.transfer_read %[[SRC_SLICE]]{{.+}}, %[[READ_MASK]]
// CHECK:           %[[TRANSP:.+]] = vector.transpose %[[READ]], [0, 2, 1, 3]
// CHECK:           %[[SHAPE_CAST:.+]] = vector.shape_cast %[[TRANSP]] : vector<1x16x2x16xf32> to vector<16x32xf32>
// CHECK:           %[[WRITE_MASK:.+]] = vector.create_mask %[[DEST_SZ0]], %[[DEST_SZ1]] : vector<16x32xi1>
// CHECK:           %[[UNPACK_WRITE:.+]] = vector.transfer_write %[[SHAPE_CAST]], {{.+}}, %[[WRITE_MASK]]
// CHECK:           %[[D0:.+]] = tensor.dim %[[UNPACK_WRITE]], %[[C0]]
// CHECK:           %[[D1:.+]] = tensor.dim %[[UNPACK_WRITE]], %[[C1]]
// CHECK:           %[[GENERIC_MASK:.+]] = vector.create_mask %[[D0]], %[[D1]] : vector<16x32xi1>
// CHECK:           %[[GENERIC_SRC:.+]] = vector.transfer_read %[[UNPACK_WRITE]]{{.+}}, %[[GENERIC_MASK]]
// CHECK:           %[[EXP:.+]] = math.exp %[[GENERIC_SRC]]
// CHECK:           vector.transfer_write %[[EXP]]{{.+}}, %[[GENERIC_MASK]]

// -----

// CHECK-LABEL: @val_defined_by_scf_for
func.func @val_defined_by_scf_for(%arg0: index, %arg1: index) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.empty() : tensor<1x1x16x16xf32>
  %1 = tensor.empty(%arg0, %arg1) : tensor<?x?xf32>
  %2 = scf.for %arg2 = %c0 to %c16 step %c2 iter_args(%arg3 = %0) -> (tensor<1x1x16x16xf32>) {
    scf.yield %arg3 : tensor<1x1x16x16xf32>
  }
  %unpack = linalg.unpack %2 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %1 : tensor<1x1x16x16xf32> -> tensor<?x?xf32>
  return %unpack : tensor<?x?xf32>
}
// CHECK: %[[EMPTY:.*]] = tensor.empty{{.*}}: tensor<?x?xf32>
// CHECK: %[[FOR:.*]] = scf.for
// CHECK: %[[READ:.*]] = vector.transfer_read %[[FOR]]{{.*}} vector<1x1x16x16xf32>
// CHECK: %[[CAST:.*]] = vector.shape_cast %[[READ]]{{.*}} vector<16x16xf32>
// CHECK: %[[MASK:.*]] = vector.create_mask
// CHECK: %[[WRITE:.*]] = vector.transfer_write %[[CAST]], %[[EMPTY]]{{.*}}, %[[MASK]]
// CHECK: return %[[WRITE]]

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

// CHECK-LABEL: func.func @dynamic_fill_with_scalable_tiling_infer_vector_size
// CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<1x1x4x[4]xf32>
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       vector.transfer_write %[[CST]], {{.*}} {in_bounds = [true, true, true, true]} : vector<1x1x4x[4]xf32>, tensor<1x1x4x?xf32>

// -----

func.func @pad_lowered_as_masked_transfer_read(%arg0: tensor<?x?xf32>, %arg1: index) -> tensor<1x4xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %padded = tensor.pad %arg0 low[0, 0] high[%arg1, 0] {
    ^bb0(%arg2: index, %arg3: index):
      tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<1x4xf32>
  return %padded : tensor<1x4xf32>
}

// CHECK-LABEL: func.func @pad_lowered_as_masked_transfer_read
// CHECK-SAME:  %[[ARG0:.*]]: tensor<?x?xf32>
// CHECK-SAME:  %[[ARG1:.*]]: index

// CHECK-DAG: %[[CST:.*]] = arith.constant 0.0
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK:     %[[DIM0:.*]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK:     %[[DIM1:.*]] = tensor.dim %[[ARG0]], %[[C1]]
// CHECK:     %[[MASK:.*]] = vector.create_mask %[[DIM0]], %[[DIM1]]
// CHECK:     %[[READ:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], %[[CST]], %[[MASK]] {in_bounds = [true, true]}
// CHECK:     %[[EMPTY:.*]] = tensor.empty()
// CHECK:     %[[WRITE:.*]] = vector.transfer_write %[[READ]], %[[EMPTY]][%[[C0]], %[[C0]]] {in_bounds = [true, true]}
// CHECK:     return %[[WRITE]]

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

// CHECK-LABEL: func.func @dynamic_fill_with_scalable_tiling_infer_remainder_vector_size
// CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<1x1x4x[4]xf32>
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       vector.transfer_write %[[CST]], {{.*}} {in_bounds = [true, true, true, true]} : vector<1x1x4x[4]xf32>, tensor<1x1x4x?xf32>

// -----

// Test that unpack operations following ukernel operations can properly infer
// vector sizes from the ukernel output shape.

func.func @ukernel_unpack_infer_vector_sizes(%lhs: tensor<1x8x16x1xf32>, %rhs: tensor<1x8x16x1xf32>, %dest: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %init = tensor.empty() : tensor<1x1x16x16xf32>
  %ukernel = iree_codegen.ukernel.generic "foo"
    ins(%lhs, %rhs : tensor<1x8x16x1xf32>, tensor<1x8x16x1xf32>)
    outs(%init : tensor<1x1x16x16xf32>)
    -> tensor<1x1x16x16xf32>
  %unpack = linalg.unpack %ukernel
    outer_dims_perm = [0, 1]
    inner_dims_pos = [0, 1]
    inner_tiles = [16, 16]
    into %dest
    : tensor<1x1x16x16xf32> -> tensor<16x16xf32>
  return %unpack : tensor<16x16xf32>
}
// CHECK-LABEL: func.func @ukernel_unpack_infer_vector_sizes
// CHECK:         %[[UKERNEL:.*]] = iree_codegen.ukernel.generic "foo"
// CHECK:         %[[READ:.*]] = vector.transfer_read %[[UKERNEL]]{{.*}} : tensor<1x1x16x16xf32>, vector<1x1x16x16xf32>
// CHECK:         %[[CAST:.*]] = vector.shape_cast %[[READ]] : vector<1x1x16x16xf32> to vector<16x16xf32>
// CHECK:         vector.transfer_write %[[CAST]]{{.*}} : vector<16x16xf32>, tensor<16x16xf32>

// -----

func.func @negative_no_vectorize_large_vector(%arg0 : tensor<1x9007199254740991x1xf16>, %output : tensor<1x9007199254740991xf32>) -> tensor<1x9007199254740991xf32> {
  %cst_2 = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 8.000000e+00 : f16
  %r = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<1x9007199254740991x1xf16>) outs(%output : tensor<1x9007199254740991xf32>) {
  ^bb0(%in: f16, %out: f32):
    %76 = arith.truncf %cst_2 : f32 to f16
    %77 = arith.divf %in, %cst_0 : f16
    %78 = arith.addf %77, %76 : f16
    %79 = arith.extf %78 : f16 to f32
    %80 = arith.maxnumf %79, %out : f32
    linalg.yield %80 : f32
  } -> tensor<1x9007199254740991xf32>
  return %r : tensor<1x9007199254740991xf32>
}

// CHECK-LABEL:   func.func @negative_no_vectorize_large_vector(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<1x9007199254740991x1xf16>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<1x9007199254740991xf32>) -> tensor<1x9007199254740991xf32> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0.000000e+00 : f16
// CHECK:           %[[VAL_1:.*]] = arith.constant 8.000000e+00 : f16
// CHECK:           %[[VAL_2:.*]] = linalg.generic {indexing_maps = [#{{.*}}, #{{.*}}], iterator_types = ["parallel", "parallel", "reduction"]} ins(%[[ARG0]] : tensor<1x9007199254740991x1xf16>) outs(%[[ARG1]] : tensor<1x9007199254740991xf32>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f16, %[[VAL_4:.*]]: f32):
// CHECK:             %[[VAL_5:.*]] = arith.divf %[[VAL_3]], %[[VAL_1]] : f16
// CHECK:             %[[VAL_6:.*]] = arith.addf %[[VAL_5]], %[[VAL_0]] : f16
// CHECK:             %[[VAL_7:.*]] = arith.extf %[[VAL_6]] : f16 to f32
// CHECK:             %[[VAL_8:.*]] = arith.maxnumf %[[VAL_7]], %[[VAL_4]] : f32
// CHECK:             linalg.yield %[[VAL_8]] : f32
// CHECK:           } -> tensor<1x9007199254740991xf32>
// CHECK:           return %[[VAL_2]] : tensor<1x9007199254740991xf32>
// CHECK:         }

// -----

func.func @linalg_ext_gather(%source : tensor<1024x128xi32>, %indices : tensor<10xi32>) -> (tensor<10x128xi32>) {
  %empty = tensor.empty() : tensor<10x128xi32>
  %result = iree_linalg_ext.gather dimension_map = [0]
                          ins(%source, %indices : tensor<1024x128xi32>, tensor<10xi32>)
                          outs(%empty: tensor<10x128xi32>) -> tensor<10x128xi32>
  return %result : tensor<10x128xi32>
}

// CHECK-LABEL: @linalg_ext_gather
//       CHECK:   transfer_gather

// -----

func.func @arg_compare_implicit_index(%input: tensor<4x128xf32>,
                                      %out_val: tensor<4xf32>,
                                      %out_idx: tensor<4xi32>) -> (tensor<4xf32>, tensor<4xi32>) {
  %result:2 = iree_linalg_ext.arg_compare
    dimension(1)
    ins(%input : tensor<4x128xf32>)
    outs(%out_val, %out_idx : tensor<4xf32>, tensor<4xi32>) {
  ^bb0(%a: f32, %b: f32):
    %cmp = arith.cmpf ogt, %a, %b : f32
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<4xf32>, tensor<4xi32>
  return %result#0, %result#1 : tensor<4xf32>, tensor<4xi32>
}
// CHECK-LABEL: func.func @arg_compare_implicit_index
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[OUT_VAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[OUT_IDX:[a-zA-Z0-9]+]]
// CHECK:         %[[INPUT_VEC:.+]] = vector.transfer_read %[[INPUT]]
// CHECK-SAME:      tensor<4x128xf32>, vector<4x128xf32>
// CHECK:         %[[OUT_VAL_VEC:.+]] = vector.transfer_read %[[OUT_VAL]]
// CHECK-SAME:      tensor<4xf32>, vector<4xf32>
// CHECK:         %[[OUT_IDX_VEC:.+]] = vector.transfer_read %[[OUT_IDX]]
// CHECK-SAME:      tensor<4xi32>, vector<4xi32>
// CHECK:         %[[RESULT_VAL:.+]], %[[RESULT_IDX:.+]] = iree_vector_ext.arg_compare
// CHECK-SAME:      dimension(1)
// CHECK-SAME:      ins(%[[INPUT_VEC]] : vector<4x128xf32>)
// CHECK-SAME:      inits(%[[OUT_VAL_VEC]], %[[OUT_IDX_VEC]] : vector<4xf32>, vector<4xi32>)
// CHECK:         ^bb0(%[[A:.+]]: f32, %[[B:.+]]: f32):
// CHECK:           %[[CMP:.+]] = arith.cmpf ogt, %[[A]], %[[B]] : f32
// CHECK:           iree_vector_ext.yield %[[CMP]] : i1
// CHECK:         -> vector<4xf32>, vector<4xi32>
// CHECK:         %[[WRITE_VAL:.+]] = vector.transfer_write %[[RESULT_VAL]], %[[OUT_VAL]]
// CHECK-SAME:      vector<4xf32>, tensor<4xf32>
// CHECK:         %[[WRITE_IDX:.+]] = vector.transfer_write %[[RESULT_IDX]], %[[OUT_IDX]]
// CHECK-SAME:      vector<4xi32>, tensor<4xi32>
// CHECK:         return %[[WRITE_VAL]], %[[WRITE_IDX]]

// -----

func.func @arg_compare_explicit_index(%partial_vals: tensor<4x32xf32>,
                                      %partial_idxs: tensor<4x32xi32>,
                                      %out_val: tensor<4xf32>,
                                      %out_idx: tensor<4xi32>) -> (tensor<4xf32>, tensor<4xi32>) {
  %result:2 = iree_linalg_ext.arg_compare
    dimension(1)
    ins(%partial_vals, %partial_idxs : tensor<4x32xf32>, tensor<4x32xi32>)
    outs(%out_val, %out_idx : tensor<4xf32>, tensor<4xi32>) {
  ^bb0(%a: f32, %b: f32):
    %cmp = arith.cmpf ogt, %a, %b : f32
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<4xf32>, tensor<4xi32>
  return %result#0, %result#1 : tensor<4xf32>, tensor<4xi32>
}
// CHECK-LABEL: func.func @arg_compare_explicit_index
// CHECK-SAME:    %[[PARTIAL_VALS:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[PARTIAL_IDXS:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[OUT_VAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[OUT_IDX:[a-zA-Z0-9]+]]
// CHECK:         %[[VALS_VEC:.+]] = vector.transfer_read %[[PARTIAL_VALS]]
// CHECK-SAME:      tensor<4x32xf32>, vector<4x32xf32>
// CHECK:         %[[IDXS_VEC:.+]] = vector.transfer_read %[[PARTIAL_IDXS]]
// CHECK-SAME:      tensor<4x32xi32>, vector<4x32xi32>
// CHECK:         %[[OUT_VAL_VEC:.+]] = vector.transfer_read %[[OUT_VAL]]
// CHECK-SAME:      tensor<4xf32>, vector<4xf32>
// CHECK:         %[[OUT_IDX_VEC:.+]] = vector.transfer_read %[[OUT_IDX]]
// CHECK-SAME:      tensor<4xi32>, vector<4xi32>
// CHECK:         %[[RESULT_VAL:.+]], %[[RESULT_IDX:.+]] = iree_vector_ext.arg_compare
// CHECK-SAME:      dimension(1)
// CHECK-SAME:      ins(%[[VALS_VEC]], %[[IDXS_VEC]] : vector<4x32xf32>, vector<4x32xi32>)
// CHECK-SAME:      inits(%[[OUT_VAL_VEC]], %[[OUT_IDX_VEC]] : vector<4xf32>, vector<4xi32>)
// CHECK:         ^bb0(%[[A:.+]]: f32, %[[B:.+]]: f32):
// CHECK:           %[[CMP:.+]] = arith.cmpf ogt, %[[A]], %[[B]] : f32
// CHECK:           iree_vector_ext.yield %[[CMP]] : i1
// CHECK:         -> vector<4xf32>, vector<4xi32>
// CHECK:         %[[WRITE_VAL:.+]] = vector.transfer_write %[[RESULT_VAL]], %[[OUT_VAL]]
// CHECK-SAME:      vector<4xf32>, tensor<4xf32>
// CHECK:         %[[WRITE_IDX:.+]] = vector.transfer_write %[[RESULT_IDX]], %[[OUT_IDX]]
// CHECK-SAME:      vector<4xi32>, tensor<4xi32>
// CHECK:         return %[[WRITE_VAL]], %[[WRITE_IDX]]

// -----

func.func @arg_compare_with_index_base(%input: tensor<4x128xf32>,
                                       %out_val: tensor<4xf32>,
                                       %out_idx: tensor<4xi32>) -> (tensor<4xf32>, tensor<4xi32>) {
  %base = arith.constant 64 : index
  %result:2 = iree_linalg_ext.arg_compare
    dimension(1)
    ins(%input : tensor<4x128xf32>)
    outs(%out_val, %out_idx : tensor<4xf32>, tensor<4xi32>)
    index_base(%base : index) {
  ^bb0(%a: f32, %b: f32):
    %cmp = arith.cmpf ogt, %a, %b : f32
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<4xf32>, tensor<4xi32>
  return %result#0, %result#1 : tensor<4xf32>, tensor<4xi32>
}
// CHECK-LABEL: func.func @arg_compare_with_index_base
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[OUT_VAL:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[OUT_IDX:[a-zA-Z0-9]+]]
// CHECK:         %[[BASE:.+]] = arith.constant 64 : index
// CHECK:         %[[INPUT_VEC:.+]] = vector.transfer_read %[[INPUT]]
// CHECK:         %[[OUT_VAL_VEC:.+]] = vector.transfer_read %[[OUT_VAL]]
// CHECK:         %[[OUT_IDX_VEC:.+]] = vector.transfer_read %[[OUT_IDX]]
// CHECK:         %[[RESULT_VAL:.+]], %[[RESULT_IDX:.+]] = iree_vector_ext.arg_compare
// CHECK-SAME:      dimension(1)
// CHECK-SAME:      ins(%[[INPUT_VEC]] : vector<4x128xf32>)
// CHECK-SAME:      inits(%[[OUT_VAL_VEC]], %[[OUT_IDX_VEC]] : vector<4xf32>, vector<4xi32>)
// CHECK-SAME:      index_base(%[[BASE]] : index)
// CHECK:         ^bb0(%[[A:.+]]: f32, %[[B:.+]]: f32):
// CHECK:           %[[CMP:.+]] = arith.cmpf ogt, %[[A]], %[[B]] : f32
// CHECK:           iree_vector_ext.yield %[[CMP]] : i1
// CHECK:         -> vector<4xf32>, vector<4xi32>
// CHECK:         %[[WRITE_VAL:.+]] = vector.transfer_write %[[RESULT_VAL]], %[[OUT_VAL]]
// CHECK:         %[[WRITE_IDX:.+]] = vector.transfer_write %[[RESULT_IDX]], %[[OUT_IDX]]
// CHECK:         return %[[WRITE_VAL]], %[[WRITE_IDX]]

// -----

// Test 1D reduction with rank-reducing extract_slice (1D -> 0D).
// This covers the failure case where output is scalar from extract_slice.
func.func @arg_compare_1d_reduction_extract_slice(%input: tensor<1024xf32>,
                                                  %init_val_tile: tensor<1xf32>,
                                                  %init_idx_tile: tensor<1xi64>)
    -> (tensor<f32>, tensor<i64>) {
  %out_val = tensor.extract_slice %init_val_tile[0] [1] [1] : tensor<1xf32> to tensor<f32>
  %out_idx = tensor.extract_slice %init_idx_tile[0] [1] [1] : tensor<1xi64> to tensor<i64>
  %result:2 = iree_linalg_ext.arg_compare
    dimension(0)
    ins(%input : tensor<1024xf32>)
    outs(%out_val, %out_idx : tensor<f32>, tensor<i64>) {
  ^bb0(%a: f32, %b: f32):
    %cmp = arith.cmpf ogt, %a, %b : f32
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<f32>, tensor<i64>
  return %result#0, %result#1 : tensor<f32>, tensor<i64>
}
// CHECK-LABEL: func.func @arg_compare_1d_reduction_extract_slice
// CHECK:         %[[OUT_VAL:.+]] = tensor.extract_slice
// CHECK:         %[[OUT_IDX:.+]] = tensor.extract_slice
// CHECK:         %[[INPUT_VEC:.+]] = vector.transfer_read
// CHECK-SAME:      tensor<1024xf32>, vector<1024xf32>
// CHECK:         %[[OUT_VAL_VEC:.+]] = vector.transfer_read %[[OUT_VAL]]
// CHECK-SAME:      tensor<f32>, vector<f32>
// CHECK:         %[[OUT_IDX_VEC:.+]] = vector.transfer_read %[[OUT_IDX]]
// CHECK-SAME:      tensor<i64>, vector<i64>
// CHECK:         %[[RESULT_VAL:.+]], %[[RESULT_IDX:.+]] = iree_vector_ext.arg_compare
// CHECK-SAME:      dimension(0)
// CHECK-SAME:      ins(%[[INPUT_VEC]] : vector<1024xf32>)
// CHECK-SAME:      inits(%[[OUT_VAL_VEC]], %[[OUT_IDX_VEC]] : vector<f32>, vector<i64>)
// CHECK:         -> vector<f32>, vector<i64>

// -----

// Test multi-dimensional batched reduction where output comes from extract_slice
// (simulates scf.forall batching). The fix ensures vectorSizes comes from input
// shape [4, 8, 128] even when inferSizesFromIR would return wrong sizes [4, 8, 1]
// from extract_slice.
func.func @arg_compare_multidim_batched_extract_slice(%input: tensor<4x8x128xf32>,
                                                      %out_val_tile: tensor<4x8x1xf32>,
                                                      %out_idx_tile: tensor<4x8x1xi32>)
    -> (tensor<4x8xf32>, tensor<4x8xi32>) {
  %out_val = tensor.extract_slice %out_val_tile[0, 0, 0] [4, 8, 1] [1, 1, 1] : tensor<4x8x1xf32> to tensor<4x8xf32>
  %out_idx = tensor.extract_slice %out_idx_tile[0, 0, 0] [4, 8, 1] [1, 1, 1] : tensor<4x8x1xi32> to tensor<4x8xi32>
  %result:2 = iree_linalg_ext.arg_compare
    dimension(2)
    ins(%input : tensor<4x8x128xf32>)
    outs(%out_val, %out_idx : tensor<4x8xf32>, tensor<4x8xi32>) {
  ^bb0(%a: f32, %b: f32):
    %cmp = arith.cmpf ogt, %a, %b : f32
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<4x8xf32>, tensor<4x8xi32>
  return %result#0, %result#1 : tensor<4x8xf32>, tensor<4x8xi32>
}
// CHECK-LABEL: func.func @arg_compare_multidim_batched_extract_slice
// CHECK:         %[[OUT_VAL:.+]] = tensor.extract_slice
// CHECK:         %[[OUT_IDX:.+]] = tensor.extract_slice
// CHECK:         %[[INPUT_VEC:.+]] = vector.transfer_read
// CHECK-SAME:      tensor<4x8x128xf32>, vector<4x8x128xf32>
// CHECK:         %[[OUT_VAL_VEC:.+]] = vector.transfer_read %[[OUT_VAL]]
// CHECK-SAME:      tensor<4x8xf32>, vector<4x8xf32>
// CHECK:         %[[OUT_IDX_VEC:.+]] = vector.transfer_read %[[OUT_IDX]]
// CHECK-SAME:      tensor<4x8xi32>, vector<4x8xi32>
// CHECK:         %[[RESULT_VAL:.+]], %[[RESULT_IDX:.+]] = iree_vector_ext.arg_compare
// CHECK-SAME:      dimension(2)
// CHECK-SAME:      ins(%[[INPUT_VEC]] : vector<4x8x128xf32>)
// CHECK-SAME:      inits(%[[OUT_VAL_VEC]], %[[OUT_IDX_VEC]] : vector<4x8xf32>, vector<4x8xi32>)
// CHECK:         -> vector<4x8xf32>, vector<4x8xi32>

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [64, 64],

  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

func.func @vectorize_to_layout(%A: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %AL = iree_vector_ext.to_layout %A to layout(#layout) : tensor<64x64xf32>
  return %AL : tensor<64x64xf32>
}

// CHECK-LABEL: func.func @vectorize_to_layout
// CHECK-SAME: %[[AT:.+]]: tensor<64x64xf32>
// CHECK: %[[A_READ:.+]] = vector.transfer_read %[[AT]]
// CHECK: %[[A:.+]] = iree_vector_ext.to_layout %[[A_READ]]
// CHECK: %[[A_WRITE:.+]] = vector.transfer_write %[[A]], %[[AT]]

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [4, 2],
  outer_tile = [1, 1],
  thread_tile = [8, 4],
  element_tile = [8, 8],

  subgroup_strides = [0, 0],
  thread_strides   = [4, 1]
>

func.func @vectorize_to_layout_with_mask(%A: tensor<256x63xf32>) -> tensor<256x63xf32> {
  %AL = iree_vector_ext.to_layout %A to layout(#layout) : tensor<256x63xf32>
  return %AL : tensor<256x63xf32>
}

// CHECK-LABEL: func.func @vectorize_to_layout_with_mask
// CHECK-SAME: %[[AT:.+]]: tensor<256x63xf32>
// CHECK: %[[MASK:.+]] = vector.constant_mask [256, 63]
// CHECK: %[[A_READ:.+]] = vector.transfer_read %[[AT]]{{.*}} %[[MASK]]
// CHECK: %[[A:.+]] = iree_vector_ext.to_layout %[[A_READ]]
// CHECK: %[[A_WRITE:.+]] = vector.transfer_write %[[A]], %[[AT]]{{.*}} %[[MASK]]

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @tensor_multi_mma(%lhs: tensor<2x3x4xf16>, %rhs: tensor<3x5x4xf16>, %acc: tensor<2x5x4xf32>) -> tensor<2x5x4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : tensor<2x3x4xf16>, tensor<3x5x4xf16> into tensor<2x5x4xf32>
  return %0 : tensor<2x5x4xf32>
}

//      CHECK-LABEL: func @tensor_multi_mma

//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.000000e+00 : f16
//   CHECK-DAG:   %[[CSTF32:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[LHS:.+]] = vector.transfer_read %arg0[%c0, %c0, %c0], %[[CST]] {{.*}} : tensor<2x3x4xf16>, vector<2x3x4xf16>
//   CHECK-DAG:   %[[RHS:.+]] = vector.transfer_read %arg1[%c0, %c0, %c0], %[[CST]] {{.*}} : tensor<3x5x4xf16>, vector<3x5x4xf16>
//   CHECK-DAG:   %[[ACC:.+]] = vector.transfer_read %arg2[%c0, %c0, %c0], %[[CSTF32]] {{.*}} : tensor<2x5x4xf32>, vector<2x5x4xf32>
//       CHECK:   %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
//  CHECK-SAME:     : vector<2x3x4xf16>, vector<3x5x4xf16> into vector<2x5x4xf32>
//       CHECK:   vector.transfer_write %[[MMA]], %arg2[%c0, %c0, %c0] {{.*}} : vector<2x5x4xf32>, tensor<2x5x4xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @tensor_single_multi_mma(%lhs: tensor<4xf16>, %rhs: tensor<4xf16>, %acc: tensor<4xf32>) -> tensor<4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : tensor<4xf16>, tensor<4xf16> into tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @tensor_single_multi_mma

//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.000000e+00 : f16
//   CHECK-DAG:   %[[CSTF32:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[LHS:.+]] = vector.transfer_read %arg0[%c0], %[[CST]] {in_bounds = [true]} : tensor<4xf16>, vector<4xf16>
//   CHECK-DAG:   %[[RHS:.+]] = vector.transfer_read %arg1[%c0], %[[CST]] {in_bounds = [true]} : tensor<4xf16>, vector<4xf16>
//   CHECK-DAG:   %[[ACC:.+]] = vector.transfer_read %arg2[%c0], %[[CSTF32]] {in_bounds = [true]} : tensor<4xf32>, vector<4xf32>
//       CHECK:   %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
//  CHECK-SAME:     : vector<4xf16>, vector<4xf16> into vector<4xf32>
//       CHECK:   vector.transfer_write %[[MMA]], %arg2[%c0] {in_bounds = [true]} : vector<4xf32>, tensor<4xf32>

// -----

#contraction_accesses = [
 affine_map<(i, j, k, b) -> (i, k, b)>,
 affine_map<(i, j, k, b) -> (k, b, j)>,
 affine_map<(i, j, k, b) -> (i, k)>,
 affine_map<(i, j, k, b) -> (k, j)>,
 affine_map<(i, j, k, b) -> (i, j)>
]

#iterator_types = [
  #linalg.iterator_type<parallel>,
  #linalg.iterator_type<parallel>,
  #linalg.iterator_type<reduction>,
  #linalg.iterator_type<reduction>
]

func.func @scaled_tensor_multi_mma(%arg0: tensor<3x5x1x32xf4E2M1FN>, %arg1: tensor<5x1x7x32xf8E4M3FN>, %arg2: tensor<3x5x1xf8E8M0FNU>, %arg3: tensor<5x7x1xf8E8M0FNU>,
  %arg4: tensor<3x7x4xf32>) -> tensor<3x7x4xf32> {
  %0 = iree_codegen.inner_tiled ins(%arg0, %arg1, %arg2, %arg3) outs(%arg4) {
    indexing_maps = #contraction_accesses,
    iterator_types = #iterator_types,
    kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32,
      lhs_elem_type = f4E2M1FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
    } : tensor<3x5x1x32xf4E2M1FN>, tensor<5x1x7x32xf8E4M3FN>, tensor<3x5x1xf8E8M0FNU>, tensor<5x7x1xf8E8M0FNU>
      into tensor<3x7x4xf32>
  return %0 : tensor<3x7x4xf32>
}

// CHECK-LABEL: func @scaled_tensor_multi_mma

//   CHECK-DAG:   %[[CSTFP4:.+]] = arith.constant 0.000000e+00 : f4E2M1FN
//   CHECK-DAG:   %[[CSTFP8:.+]] = arith.constant 0.000000e+00 : f8E4M3FN
//   CHECK-DAG:   %[[CSTSCALE:.+]] = arith.constant 5.877470e-39 : f8E8M0FNU
//   CHECK-DAG:   %[[CSTF32:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[LHS:.+]] = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %[[CSTFP4]] {{.*}} : tensor<3x5x1x32xf4E2M1FN>, vector<3x5x1x32xf4E2M1FN>
//   CHECK-DAG:   %[[RHS:.+]] = vector.transfer_read %arg1[%c0, %c0, %c0, %c0], %[[CSTFP8]] {{.*}} : tensor<5x1x7x32xf8E4M3FN>, vector<5x1x7x32xf8E4M3FN>
//   CHECK-DAG:   %[[LHS_SCALE:.+]] = vector.transfer_read %arg2[%c0, %c0, %c0], %[[CSTSCALE]] {{.*}} : tensor<3x5x1xf8E8M0FNU>, vector<3x5x1xf8E8M0FNU>
//   CHECK-DAG:   %[[RHS_SCALE:.+]] = vector.transfer_read %arg3[%c0, %c0, %c0], %[[CSTSCALE]] {{.*}} : tensor<5x7x1xf8E8M0FNU>, vector<5x7x1xf8E8M0FNU>
//   CHECK-DAG:   %[[ACC:.+]] = vector.transfer_read %arg4[%c0, %c0, %c0], %[[CSTF32]] {{.*}} : tensor<3x7x4xf32>, vector<3x7x4xf32>
//       CHECK:   %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]], %[[LHS_SCALE]], %[[RHS_SCALE]]) outs(%[[ACC]])
//  CHECK-SAME: : vector<3x5x1x32xf4E2M1FN>, vector<5x1x7x32xf8E4M3FN>, vector<3x5x1xf8E8M0FNU>, vector<5x7x1xf8E8M0FNU> into vector<3x7x4xf32>
//       CHECK:   vector.transfer_write %[[MMA]], %arg4[%c0, %c0, %c0] {{.*}} : vector<3x7x4xf32>, tensor<3x7x4xf32>

// -----

// Masked vectorization of inner_tiled with dynamic outer dimensions.
// Vector sizes come from the iree_codegen.vector_tile_sizes attribute.
// The LHS has shape <?x?x4xf16> (outer dims dynamic), iteration space is
// [i=2, j=5, k=3], so reads should be masked.
#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @masked_tensor_multi_mma(%lhs: tensor<?x?x4xf16>, %rhs: tensor<?x?x4xf16>, %acc: tensor<?x?x4xf32>) -> tensor<?x?x4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    iree_codegen.vector_tile_sizes = array<i64: 2, 5, 3>,
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : tensor<?x?x4xf16>, tensor<?x?x4xf16> into tensor<?x?x4xf32>
  return %0 : tensor<?x?x4xf32>
}

// CHECK-LABEL: func @masked_tensor_multi_mma

// With vectorSizes, reads use create_mask for dynamic dims.
// LHS: outer (i=2, k=3) + inner (4) → vector<2x3x4xf16>
// RHS: outer (k=3, j=5) + inner (4) → vector<3x5x4xf16>
// ACC: outer (i=2, j=5) + inner (4) → vector<2x5x4xf32>
//   CHECK-DAG:   %[[CSTF16:.+]] = arith.constant 0.000000e+00 : f16
//   CHECK-DAG:   %[[CSTF32:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[LHS_MASK:.+]] = vector.create_mask {{.*}} : vector<2x3x4xi1>
//       CHECK:   %[[LHS:.+]] = vector.transfer_read %arg0{{.*}}, %[[CSTF16]], %[[LHS_MASK]]{{.*}} : tensor<?x?x4xf16>, vector<2x3x4xf16>
//       CHECK:   %[[RHS_MASK:.+]] = vector.create_mask {{.*}} : vector<3x5x4xi1>
//       CHECK:   %[[RHS:.+]] = vector.transfer_read %arg1{{.*}}, %[[CSTF16]], %[[RHS_MASK]]{{.*}} : tensor<?x?x4xf16>, vector<3x5x4xf16>
//       CHECK:   %[[ACC_MASK:.+]] = vector.create_mask {{.*}} : vector<2x5x4xi1>
//       CHECK:   %[[ACC:.+]] = vector.transfer_read %arg2{{.*}}, %[[CSTF32]], %[[ACC_MASK]]{{.*}} : tensor<?x?x4xf32>, vector<2x5x4xf32>
//       CHECK:   %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
//  CHECK-SAME:     : vector<2x3x4xf16>, vector<3x5x4xf16> into vector<2x5x4xf32>
//       CHECK:   %[[WRITE_MASK:.+]] = vector.create_mask {{.*}} : vector<2x5x4xi1>
//       CHECK:   vector.transfer_write %[[MMA]], %arg2{{.*}}, %[[WRITE_MASK]] {in_bounds = [false, false, true]}
//  CHECK-SAME:     : vector<2x5x4xf32>, tensor<?x?x4xf32>

// -----

// Tests for im2col op vectorization via VectorizableOpInterface.

// Standard NHWC layout, K tile size (4) divides innermost input dim C (640).
// Vectorizes along K (output dim 2) with vector width 4.
// Non-vectorized dims: batch (2) x M (2) = 4 iterations.
#im2col_map_k = affine_map<(d0) -> (d0 * 4)>
func.func @im2col_vectorize_nhwc(
    %input: tensor<2x34x34x640xf32>, %m_off: index, %k: index
) -> tensor<2x2x4xf32> {
  %0 = tensor.empty() : tensor<2x2x4xf32>
  %k_off = affine.apply #im2col_map_k(%k)
  %1 = iree_linalg_ext.im2col
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          offsets = [0, %m_off, %k_off] output_sizes = [[2], [32, 32], [3, 3, 640]]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
          ins(%input : tensor<2x34x34x640xf32>)
          outs(%0 : tensor<2x2x4xf32>) -> tensor<2x2x4xf32>
  return %1 : tensor<2x2x4xf32>
}
// CHECK-LABEL: func.func @im2col_vectorize_nhwc
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<2x34x34x640xf32>
//   CHECK-DAG:   %[[POISON:.+]] = ub.poison : f32
//   CHECK-NOT:   iree_linalg_ext.im2col
//       CHECK:   %[[R0:.+]] = vector.transfer_read %[[INPUT]]{{.*}}, %[[POISON]] {in_bounds = [true]} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write %[[R0]], {{.*}} : vector<4xf32>, tensor<2x2x4xf32>
//       CHECK:   %[[R1:.+]] = vector.transfer_read %[[INPUT]]{{.*}}, %[[POISON]] {in_bounds = [true]} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write %[[R1]], {{.*}} : vector<4xf32>, tensor<2x2x4xf32>
//       CHECK:   %[[R2:.+]] = vector.transfer_read %[[INPUT]]{{.*}}, %[[POISON]] {in_bounds = [true]} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write %[[R2]], {{.*}} : vector<4xf32>, tensor<2x2x4xf32>
//       CHECK:   %[[R3:.+]] = vector.transfer_read %[[INPUT]]{{.*}}, %[[POISON]] {in_bounds = [true]} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   %[[FINAL:.+]] = vector.transfer_write %[[R3]], {{.*}} : vector<4xf32>, tensor<2x2x4xf32>
//       CHECK:   return %[[FINAL]] : tensor<2x2x4xf32>

// -----

// Dynamic output shape: vectorization pattern should not match.
func.func @im2col_no_vectorize_dynamic(
    %input: tensor<2x34x34x640xf32>, %m_size: index, %m_off: index, %k: index
) -> tensor<2x?x4xf32> {
  %0 = tensor.empty(%m_size) : tensor<2x?x4xf32>
  %k_off = affine.apply affine_map<(d0) -> (d0 * 4)>(%k)
  %1 = iree_linalg_ext.im2col
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          offsets = [0, %m_off, %k_off] output_sizes = [[2], [32, 32], [3, 3, 640]]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
          ins(%input : tensor<2x34x34x640xf32>)
          outs(%0 : tensor<2x?x4xf32>) -> tensor<2x?x4xf32>
  return %1 : tensor<2x?x4xf32>
}
// CHECK-LABEL: func.func @im2col_no_vectorize_dynamic
//       CHECK:   iree_linalg_ext.im2col
//   CHECK-NOT:   vector.transfer_read
//   CHECK-NOT:   vector.transfer_write

// -----

// Source padding (conv padding folded into im2col). NHWC layout.
// Vectorizes along K with masked transfer_read.
#im2col_map_k_pad = affine_map<(d0) -> (d0 * 4)>
func.func @im2col_vectorize_source_padding(
    %input: tensor<2x34x34x640xf32>, %m_off: index, %k: index
) -> tensor<2x2x4xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<2x2x4xf32>
  %k_off = affine.apply #im2col_map_k_pad(%k)
  %1 = iree_linalg_ext.im2col
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          offsets = [0, %m_off, %k_off] output_sizes = [[2], [34, 34], [3, 3, 640]]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
          input_pad_low = [0, 1, 1, 0] input_pad_high = [0, 1, 1, 0]
          pad_value(%cst : f32)
          ins(%input : tensor<2x34x34x640xf32>)
          outs(%0 : tensor<2x2x4xf32>) -> tensor<2x2x4xf32>
  return %1 : tensor<2x2x4xf32>
}
// CHECK-LABEL: func.func @im2col_vectorize_source_padding
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<2x34x34x640xf32>
//   CHECK-DAG:   %[[PAD:.+]] = arith.constant 0.0{{.*}} : f32
//   CHECK-NOT:   iree_linalg_ext.im2col
//       CHECK:   %[[MASK0:.+]] = vector.create_mask {{.*}} : vector<4xi1>
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}}, %[[PAD]], %[[MASK0]] {{.*}} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<4xf32>, tensor<2x2x4xf32>
//       CHECK:   %[[MASK1:.+]] = vector.create_mask {{.*}} : vector<4xi1>
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}}, %[[PAD]], %[[MASK1]] {{.*}} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<4xf32>, tensor<2x2x4xf32>
//       CHECK:   %[[MASK2:.+]] = vector.create_mask {{.*}} : vector<4xi1>
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}}, %[[PAD]], %[[MASK2]] {{.*}} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<4xf32>, tensor<2x2x4xf32>
//       CHECK:   %[[MASK3:.+]] = vector.create_mask {{.*}} : vector<4xi1>
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}}, %[[PAD]], %[[MASK3]] {{.*}} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   %[[FINAL:.+]] = vector.transfer_write {{.*}} : vector<4xf32>, tensor<2x2x4xf32>
//       CHECK:   return %[[FINAL]] : tensor<2x2x4xf32>

// -----

// Non-vectorizable due to input_k_perm = [1, 0] making innermost K
// non-contiguous in input. Falls back to scalar unrolling (vector<1>).
func.func @im2col_scalar_fallback(
    %input: tensor<1x3x2xf32>
) -> tensor<1x2x4xf32> {
  %0 = tensor.empty() : tensor<1x2x4xf32>
  %1 = iree_linalg_ext.im2col strides = [1] dilations = [1] kernel_size = [2]
                          offsets = [0, 0, 0] output_sizes = [[1], [2], [2, 2]]
                          batch_pos = [0] m_pos = [1] k_pos = [2]
                          input_k_perm = [1, 0] output_perm = [0, 1, 2]
                          ins(%input : tensor<1x3x2xf32>)
                          outs(%0 : tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
  return %1 : tensor<1x2x4xf32>
}
// CHECK-LABEL: func.func @im2col_scalar_fallback
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<1x3x2xf32>
//   CHECK-NOT:   iree_linalg_ext.im2col
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}} : tensor<1x3x2xf32>, vector<1xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<1xf32>, tensor<1x2x4xf32>

// -----

// High-side input padding on the vectorized input dimension (channels).
// Verifies: masked vector transfer_read with pad_value, im2col fully lowered.
func.func @im2col_vectorize_channel_pad_high(
    %input: tensor<59x91x16x56xbf16>, %output: tensor<1x1x1x8xbf16>,
    %off0: index
) -> tensor<1x1x1x8xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %c5 = arith.constant 5 : index
  %c3 = arith.constant 3 : index
  %c100 = arith.constant 100 : index
  %result = iree_linalg_ext.im2col
      strides = [1, 1] dilations = [1, 1] kernel_size = [59, 91]
      offsets = [%off0, %c3, %c5, %c100]
      output_sizes = [[64], [16], [3, 3], [59, 91]]
      batch_pos = [3, 2] m_pos = [0, 1] k_pos = []
      input_k_perm = [0, 1] output_perm = [2, 3, 1, 0]
      input_pad_low = [1, 1, 0, 0] input_pad_high = [1, 1, 0, 8]
      pad_value(%cst : bf16)
      ins(%input : tensor<59x91x16x56xbf16>)
      outs(%output : tensor<1x1x1x8xbf16>) -> tensor<1x1x1x8xbf16>
  return %result : tensor<1x1x1x8xbf16>
}
// CHECK-LABEL: func.func @im2col_vectorize_channel_pad_high
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<59x91x16x56xbf16>
//   CHECK-DAG:   %[[PAD:.+]] = arith.constant 0.0{{.*}} : bf16
//   CHECK-NOT:   iree_linalg_ext.im2col
//       CHECK:   vector.create_mask {{.*}} : vector<8xi1>
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}}, %[[PAD]], %{{.*}} {in_bounds = [true]} : tensor<59x91x16x56xbf16>, vector<8xbf16>
//       CHECK:   %[[FINAL:.+]] = vector.transfer_write {{.*}} : vector<8xbf16>, tensor<1x1x1x8xbf16>
//       CHECK:   return %[[FINAL]] : tensor<1x1x1x8xbf16>

// -----

// Low-side input padding on the vectorized input dimension: falls back to
// scalar unrolling (vector<1>) because chooseDimToVectorize returns nullopt.
func.func @im2col_scalar_fallback_channel_pad_low(
    %input: tensor<59x91x16x56xbf16>, %output: tensor<1x1x1x8xbf16>
) -> tensor<1x1x1x8xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %c5 = arith.constant 5 : index
  %c3 = arith.constant 3 : index
  %c42 = arith.constant 42 : index
  %c100 = arith.constant 100 : index
  %result = iree_linalg_ext.im2col
      strides = [1, 1] dilations = [1, 1] kernel_size = [59, 91]
      offsets = [%c42, %c3, %c5, %c100]
      output_sizes = [[64], [16], [3, 3], [59, 91]]
      batch_pos = [3, 2] m_pos = [0, 1] k_pos = []
      input_k_perm = [0, 1] output_perm = [2, 3, 1, 0]
      input_pad_low = [1, 1, 0, 8] input_pad_high = [1, 1, 0, 0]
      pad_value(%cst : bf16)
      ins(%input : tensor<59x91x16x56xbf16>)
      outs(%output : tensor<1x1x1x8xbf16>) -> tensor<1x1x1x8xbf16>
  return %result : tensor<1x1x1x8xbf16>
}
// All offsets are constant and in-bounds, so masks fold away. The im2col is
// fully lowered to 8 scalar (vector<1>) transfer_read/write pairs.
// CHECK-LABEL: func.func @im2col_scalar_fallback_channel_pad_low
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<59x91x16x56xbf16>
//   CHECK-NOT:   iree_linalg_ext.im2col
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}} {in_bounds = [true]} : tensor<59x91x16x56xbf16>, vector<1xbf16>
//       CHECK:   vector.transfer_write {{.*}} : vector<1xbf16>, tensor<1x1x1x8xbf16>

// -----

// Output-only padding (GEMM alignment). Vectorizes along K with masked reads.
// The output has 16 extra M positions filled with pad_value.
func.func @im2col_vectorize_output_padding(
    %input: tensor<2x34x34x640xf32>, %m_off: index, %k: index
) -> tensor<2x2x4xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<2x2x4xf32>
  %k_off = affine.apply affine_map<(d0) -> (d0 * 4)>(%k)
  %1 = iree_linalg_ext.im2col
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          offsets = [0, %m_off, %k_off] output_sizes = [[2], [32, 32], [3, 3, 640]]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
          output_pad_low = [0, 0, 0] output_pad_high = [0, 16, 0]
          pad_value(%cst : f32)
          ins(%input : tensor<2x34x34x640xf32>)
          outs(%0 : tensor<2x2x4xf32>) -> tensor<2x2x4xf32>
  return %1 : tensor<2x2x4xf32>
}
// Vectorizes along K (dim 2) with vector width 4. The output M-dim padding
// produces arith.select between the k-dim mask and all-false for each
// non-vectorized output dim. No input padding, so reads are from the
// unpadded tensor with clamped indices.
// CHECK-LABEL: func.func @im2col_vectorize_output_padding
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<2x34x34x640xf32>
//   CHECK-DAG:   %[[PAD:.+]] = arith.constant 0.0{{.*}} : f32
//   CHECK-NOT:   iree_linalg_ext.im2col
//       CHECK:   vector.transfer_read %[[INPUT]]{{.*}}, %[[PAD]], %{{.*}} {in_bounds = [true]} : tensor<2x34x34x640xf32>, vector<4xf32>
//       CHECK:   vector.transfer_write {{.*}} {in_bounds = [true]} : vector<4xf32>, tensor<2x2x4xf32>

// -----

// Output low-padding on the vectorized dim: falls back to scalar unrolling
// because chooseDimToVectorize skips dims with non-zero output_pad_low.
func.func @im2col_scalar_fallback_output_pad_low(
    %input: tensor<2x34x34x640xf32>, %m_off: index, %k: index
) -> tensor<2x2x4xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<2x2x4xf32>
  %k_off = affine.apply affine_map<(d0) -> (d0 * 4)>(%k)
  %1 = iree_linalg_ext.im2col
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          offsets = [0, %m_off, %k_off] output_sizes = [[2], [32, 32], [3, 3, 640]]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
          output_pad_low = [0, 0, 2] output_pad_high = [0, 0, 0]
          pad_value(%cst : f32)
          ins(%input : tensor<2x34x34x640xf32>)
          outs(%0 : tensor<2x2x4xf32>) -> tensor<2x2x4xf32>
  return %1 : tensor<2x2x4xf32>
}
// Scalar fallback: output_pad_low on the K dim (dim 2) prevents vectorization.
// CHECK-LABEL: func.func @im2col_scalar_fallback_output_pad_low
//   CHECK-NOT:   iree_linalg_ext.im2col
//       CHECK:   vector.transfer_read {{.*}} : tensor<2x34x34x640xf32>, vector<1xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<1xf32>, tensor<2x2x4xf32>
