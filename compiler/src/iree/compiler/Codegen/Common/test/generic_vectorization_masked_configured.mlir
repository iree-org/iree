// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-generic-vectorization{enable-vector-masking=true use-configured-vector-sizes=true}))" --split-input-file %s | FileCheck %s

// Tests for masked vectorization with pre-configured vector sizes.
// The vector sizes are explicitly specified via lowering_config attributes
// (e.g., #iree_cpu.lowering_config<vector_common_parallel = [...]>).

#config = #iree_cpu.lowering_config<vector_common_parallel = [4, 8, 0], vector_reduction = [0, 0, 16]>
func.func @matmul_with_configured_vector(%lhs: tensor<?x?xf16>, %rhs: tensor<?x?xf16>, %acc: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result = linalg.matmul {lowering_config = #config} ins(%lhs, %rhs: tensor<?x?xf16>, tensor<?x?xf16>) outs(%acc: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result: tensor<?x?xf32>
}
// CHECK-LABEL: func.func @matmul_with_configured_vector(
// CHECK-SAME:    %[[LHS:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[RHS:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[OUT:[a-zA-Z0-9]+]]
// CHECK:         %[[LHS_MASK:.+]] = vector.create_mask {{.+}} : vector<4x16xi1>
// CHECK:         %[[LHS_VEC:.+]] = vector.transfer_read %[[LHS]]{{.+}}, %[[LHS_MASK]]
// CHECK:         %[[RHS_MASK:.+]] = vector.create_mask {{.+}} : vector<16x8xi1>
// CHECK:         %[[RHS_VEC:.+]] = vector.transfer_read %[[RHS]]{{.+}}, %[[RHS_MASK]]
// CHECK:         %[[OUT_MASK:.+]] = vector.create_mask {{.+}} : vector<4x8xi1>
// CHECK:         %[[OUT_VEC:.+]] = vector.transfer_read %[[OUT]]{{.+}}, %[[OUT_MASK]]
// CHECK:         %[[EXT_LHS:.+]] = arith.extf %[[LHS_VEC]]
// CHECK:         %[[EXT_RHS:.+]] = arith.extf %[[RHS_VEC]]
// CHECK:         vector.contract {{.+}} %[[EXT_LHS]], %[[EXT_RHS]], %[[OUT_VEC]]

// -----

#config = #iree_cpu.lowering_config<vector_common_parallel = [2, 4]>
func.func @pack_with_configured_vector(%src: tensor<?x?xi8>, %dest: tensor<?x?x16x2xi8>) -> tensor<?x?x16x2xi8> {
  %c0_i8 = arith.constant 0 : i8
  %pack = linalg.pack %src padding_value(%c0_i8 : i8) outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 2] into %dest {lowering_config = #config} : tensor<?x?xi8> -> tensor<?x?x16x2xi8>
  return %pack : tensor<?x?x16x2xi8>
}
// CHECK-LABEL: func.func @pack_with_configured_vector(
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C0_I8:.+]] = arith.constant 0 : i8
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index

// Compute mask for xfer_read:
// CHECK:         %[[DIM0:.+]] = tensor.dim %[[SRC]], %[[C0]] : tensor<?x?xi8>
// CHECK:         %[[DIM1:.+]] = tensor.dim %[[SRC]], %[[C1]] : tensor<?x?xi8>
// CHECK:         %[[READ_MASK:.+]] = vector.create_mask %[[DIM0]], %[[DIM1]] : vector<8x32xi1>

// --= read =---
// CHECK:         %[[READ_VEC:.+]] = vector.transfer_read %[[SRC]][%[[C0]], %[[C0]]], %[[C0_I8]], %[[READ_MASK]]

// --= shape_cast and transpose =---
// CHECK:         %[[CAST_VEC:.+]] = vector.shape_cast %[[READ_VEC]] : vector<8x32xi8> to vector<4x2x2x16xi8>
// CHECK:         %[[TRANSP_VEC:.+]] = vector.transpose %[[CAST_VEC]], [2, 0, 3, 1]

// Compute mask for xfer_write:
// CHECK:         %[[W_DIM0:.+]] = tensor.dim %[[DEST]], %[[C0]] : tensor<?x?x16x2xi8>
// CHECK:         %[[W_DIM1:.+]] = tensor.dim %[[DEST]], %[[C1]] : tensor<?x?x16x2xi8>
// CHECK:         %[[WRITE_MASK:.+]] = vector.create_mask %[[W_DIM0]], %[[W_DIM1]], %[[C16]], %[[C2]] : vector<2x4x16x2xi1>

// --= write =---
// CHECK:         vector.transfer_write %[[TRANSP_VEC]], %[[DEST]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]], %[[WRITE_MASK]]

// -----

#config = #iree_cpu.lowering_config<vector_common_parallel = [4, [16]], vector_reduction = [0, 0]>
func.func @vectorize_dynamic_shapes_pack_scalable_vec_and_tile_size(%src: tensor<?x?xf32>, %dest: tensor<?x?x?x2xf32>) -> tensor<?x?x?x2xf32> {
  %vs = vector.vscale
  %c16 = arith.constant 16 : index
  %tile_size = arith.muli %vs, %c16 : index
  %packed = linalg.pack %src inner_dims_pos = [1, 0] inner_tiles = [%tile_size, 2] into %dest {lowering_config = #config} : tensor<?x?xf32> -> tensor<?x?x?x2xf32>
  return %packed : tensor<?x?x?x2xf32>
}
// CHECK-LABEL: func.func @vectorize_dynamic_shapes_pack_scalable_vec_and_tile_size
// CHECK:         linalg.pack

// -----

// The pre-configured input vector size is not yet supported if there are
// scalable flags.

#config = #iree_cpu.lowering_config<vector_common_parallel = [4, [16]], vector_reduction = [0, 0]>
func.func @vectorize_dynamic_shapes_unpack_scalable_vec_and_tile_size(%dest: tensor<?x?xf32>, %src: tensor<?x?x?x2xf32>) -> tensor<?x?xf32> {
  %vs = vector.vscale
  %c16 = arith.constant 16 : index
  %tile_size = arith.muli %vs, %c16 : index
  %ret = linalg.unpack %src inner_dims_pos = [1, 0] inner_tiles = [%tile_size, 2] into %dest {lowering_config = #config} : tensor<?x?x?x2xf32> -> tensor<?x?xf32>
  return %ret : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @vectorize_dynamic_shapes_unpack_scalable_vec_and_tile_size
// CHECK:         linalg.unpack

// -----

#aarch64_sve = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve", target_triple = "aarch64-none-elf"}>
#config = #iree_cpu.lowering_config<vector_common_parallel = [1, 4, [4], 0], vector_reduction = [0, 0, 0, 3]>
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

// CHECK-LABEL: func.func @depthwise_conv_fold_away_masking
// CHECK-NOT: vector.create_mask
// CHECK-NOT: vector.constant_mask
// CHECK:     vector.fma
// CHECK-NOT: vector.create_mask
// CHECK-NOT: vector.constant_mask

// -----

// When the lowering config has zero vector sizes, the vectorizer bails out
// to the IR inference path which derives correct sizes from tensor shapes.

#config = #iree_cpu.lowering_config<vector_common_parallel = [4, 0]>
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @configured_zero_vector_size_falls_back_to_inference(
    %arg0: tensor<4x1xf32>, %arg1: tensor<4x1xf32>) -> tensor<4x1xf32> {
  %result = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]}
      {lowering_config = #config}
      ins(%arg0, %arg1 : tensor<4x1xf32>, tensor<4x1xf32>)
      outs(%arg1 : tensor<4x1xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %add = arith.addf %in, %in_0 : f32
    linalg.yield %add : f32
  } -> tensor<4x1xf32>
  return %result : tensor<4x1xf32>
}
// CHECK-LABEL: func.func @configured_zero_vector_size_falls_back_to_inference(
// CHECK:         arith.addf {{.*}} : vector<4x1xf32>

// -----

#scan_masked_config = #iree_cpu.lowering_config<vector_common_parallel = [8, 16]>

func.func @vectorize_scan_masked_configured(
    %input: tensor<?x?xf32>,
    %output: tensor<?x?xf32>,
    %accum: tensor<?xf32>) -> (tensor<?x?xf32>, tensor<?xf32>) {
  %0:2 = iree_linalg_ext.scan {lowering_config = #scan_masked_config}
      dimension(1) inclusive(true)
      ins(%input : tensor<?x?xf32>)
      outs(%output, %accum : tensor<?x?xf32>, tensor<?xf32>) {
    ^bb0(%arg0: f32, %arg1: f32):
      %sum = arith.addf %arg0, %arg1 : f32
      iree_linalg_ext.yield %sum : f32
  } -> tensor<?x?xf32>, tensor<?xf32>
  return %0#0, %0#1 : tensor<?x?xf32>, tensor<?xf32>
}
// CHECK-LABEL: func.func @vectorize_scan_masked_configured(
// CHECK:         vector.create_mask {{.*}} : vector<8x16xi1>
// CHECK:         vector.transfer_read {{.*}} : tensor<?x?xf32>, vector<8x16xf32>
// CHECK:         arith.select {{.*}} : vector<8x16xi1>, vector<8x16xf32>
// CHECK:         vector.create_mask {{.*}} : vector<8xi1>
// CHECK:         vector.transfer_read {{.*}} : tensor<?xf32>, vector<8xf32>
// CHECK:         arith.select {{.*}} : vector<8xi1>, vector<8xf32>
// CHECK:         vector.scan <add>, {{.*}} {inclusive = true, reduction_dim = 1 : i64}
// CHECK:         vector.transfer_write {{.*}} : vector<8x16xf32>, tensor<?x?xf32>
// CHECK:         vector.transfer_write {{.*}} : vector<8xf32>, tensor<?xf32>
