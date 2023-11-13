// RUN: iree-opt --iree-codegen-cpu-materialize-encoding --canonicalize --cse --split-input-file %s | FileCheck %s

func.func @matmul_lowering_i8i8i32_vmvx_ukernel() attributes {
  hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {ukernels = "all"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [i8, i8, i32]>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [i8, i8, i32]>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [i8, i8, i32]>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [i8, i8, i32]>>>{%M, %K}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [i8, i8, i32]>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [i8, i8, i32]>>>{%K, %N}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [i8, i8, i32]>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [i8, i8, i32]>>>{%M, %N}
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [i8, i8, i32]>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [i8, i8, i32]>>,
                   tensor<?x?xi8, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [i8, i8, i32]>>)
      outs(%5 : tensor<?x?xi32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [i8, i8, i32]>>)
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [i8, i8, i32]>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [i8, i8, i32]>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [i8, i8, i32]>>>{%M, %N}
  return
}

//  CHECK-DAG: #[[MAP_CEILDIV:.+]] = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
//      CHECK: func @matmul_lowering_i8i8i32_vmvx_ukernel()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//      CHECK:   %[[LHS_TILE_SIZES:.+]]:2 = iree_codegen.query_tile_sizes tensor<?x?xi8, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [i8, i8, i32]>> -> index, index
//  CHECK-DAG:   %[[LHS_OUTER_SIZE0:.+]] = affine.apply #[[MAP_CEILDIV]]()[%[[M]], %[[LHS_TILE_SIZES]]#0]
//  CHECK-DAG:   %[[LHS_OUTER_SIZE1:.+]] = affine.apply #[[MAP_CEILDIV]]()[%[[K]], %[[LHS_TILE_SIZES]]#1]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi8>>{%[[LHS_OUTER_SIZE0]], %[[LHS_OUTER_SIZE1]], %[[LHS_TILE_SIZES]]#0, %[[LHS_TILE_SIZES]]#1}
//      CHECK:   %[[RHS_TILE_SIZES:.+]]:2 = iree_codegen.query_tile_sizes tensor<?x?xi8, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [i8, i8, i32]>> -> index, index
//  CHECK-DAG:   %[[RHS_OUTER_SIZE0:.+]] = affine.apply #[[MAP_CEILDIV]]()[%[[N]], %[[RHS_TILE_SIZES]]#0]
//  CHECK-DAG:   %[[RHS_OUTER_SIZE1:.+]] = affine.apply #[[MAP_CEILDIV]]()[%[[K]], %[[RHS_TILE_SIZES]]#1]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi8>>{%[[RHS_OUTER_SIZE0]], %[[RHS_OUTER_SIZE1]], %[[RHS_TILE_SIZES]]#0, %[[RHS_TILE_SIZES]]#1}
//      CHECK:   %[[RESULT_TILE_SIZES:.+]]:2 = iree_codegen.query_tile_sizes tensor<?x?xi32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [i8, i8, i32]>> -> index, index
//  CHECK-DAG:   %[[RESULT_OUTER_SIZE0:.+]] = affine.apply #[[MAP_CEILDIV]]()[%[[M]], %[[RESULT_TILE_SIZES]]#0]
//  CHECK-DAG:   %[[RESULT_OUTER_SIZE1:.+]] = affine.apply #[[MAP_CEILDIV]]()[%[[N]], %[[RESULT_TILE_SIZES]]#1]
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x?x?xi32>>{%[[RESULT_OUTER_SIZE0]], %[[RESULT_OUTER_SIZE1]], %[[RESULT_TILE_SIZES]]#0, %[[RESULT_TILE_SIZES]]#1}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[LHS_OUTER_SIZE0]], %[[LHS_OUTER_SIZE1]], %[[LHS_TILE_SIZES]]#0, %[[LHS_TILE_SIZES]]#1], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[RHS_OUTER_SIZE0]], %[[RHS_OUTER_SIZE1]], %[[RHS_TILE_SIZES]]#0, %[[RHS_TILE_SIZES]]#1], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[RESULT_OUTER_SIZE0]], %[[RESULT_OUTER_SIZE1]], %[[RESULT_TILE_SIZES]]#0, %[[RESULT_TILE_SIZES]]#1], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[RESULT_OUTER_SIZE0]], %[[RESULT_OUTER_SIZE1]], %[[RESULT_TILE_SIZES]]#0, %[[RESULT_TILE_SIZES]]#1], strides = [1, 1, 1, 1]

// -----

#map = affine_map<()[s0] -> ((3 ceildiv s0) * s0)>
#map1 = affine_map<()[s0] -> ((1 ceildiv s0) * s0)>
func.func @fill_matmul(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index) attributes {
   hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
} {
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32], original_type = tensor<1x2xf32>>>>{%arg0, %arg1}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f32, f32, f32], original_type = tensor<2x3xf32>>>>{%arg2, %arg3}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32], original_type = tensor<1x3xf32>>>>{%arg4, %arg5}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%arg0, %arg1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32], original_type = tensor<1x2xf32>>>>{%arg0, %arg1} -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32], original_type = tensor<1x2xf32>>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%arg2, %arg3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f32, f32, f32], original_type = tensor<2x3xf32>>>>{%arg2, %arg3} -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f32, f32, f32], original_type = tensor<2x3xf32>>>
  %5 = affine.apply #map()[%arg6]
  %6 = affine.apply #map1()[%arg7]
  %7 = tensor.empty(%6, %5) : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32], original_type = tensor<1x3xf32>>>
  %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32], original_type = tensor<1x3xf32>>>) -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32], original_type = tensor<1x3xf32>>>
  %9 = linalg.matmul ins(%3, %4 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32], original_type = tensor<1x2xf32>>>, tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f32, f32, f32], original_type = tensor<2x3xf32>>>) outs(%8 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32], original_type = tensor<1x3xf32>>>) -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32], original_type = tensor<1x3xf32>>>
  flow.dispatch.tensor.store %9, %2, offsets = [0, 0], sizes = [%arg4, %arg5], strides = [1, 1] : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32], original_type = tensor<1x3xf32>>> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32], original_type = tensor<1x3xf32>>>>{%arg4, %arg5}
  return
}
//      CHECK: func.func @fill_matmul
//  CHECK-DAG:   %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<1x1x8x4xf32>>
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<1x1x8x4xf32>>
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<writeonly:tensor<1x1x8x8xf32>>
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [1, 1, 8, 4], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [1, 1, 8, 4], strides = [1, 1, 1, 1]
//      CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<1x1x8x8xf32>
//      CHECK:   %[[FILL:.+]] = linalg.fill
// CHECK-SAME:     ins(%[[ZERO]]
// CHECK-SAME:     outs(%[[EMPTY]]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:     ins(%[[LHS]], %[[RHS]]
// CHECK-SAME:     outs(%[[FILL]]
//      CHECK:  flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:    offsets = [0, 0, 0, 0], sizes = [1, 1, 8, 8], strides = [1, 1, 1, 1]

// -----

func.func @set_encoding_dynamic() attributes {
   hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %d0 = hal.interface.constant.load [0] : index
  %d1 = hal.interface.constant.load [1] : index
  %outd0 = hal.interface.constant.load [2] : index
  %outd1 = hal.interface.constant.load [3] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%d0, %d1}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>>{%outd0, %outd1}
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%d0, %d1} -> tensor<?x?xf32>
  %p0 = affine.apply affine_map<()[s0, s1] -> (-s0 + s1)>()[%d0, %outd0]
  %p1 = affine.apply affine_map<()[s0, s1] -> (-s0 + s1)>()[%d1, %outd1]
  %padded = tensor.pad %2 low[0, 0] high[%p0, %p1] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<?x?xf32>
  %3 = iree_linalg_ext.set_encoding %padded : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [%outd0, %outd1], strides = [1, 1]
      : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>
      -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>>{%outd0, %outd1}
  return
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//       CHECK: func @set_encoding_dynamic()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.0
//   CHECK-DAG:   %[[D0:.+]] = hal.interface.constant.load[0]
//   CHECK-DAG:   %[[D1:.+]] = hal.interface.constant.load[1]
//   CHECK-DAG:   %[[OUTD0:.+]] = hal.interface.constant.load[2]
//   CHECK-DAG:   %[[OUTD1:.+]] = hal.interface.constant.load[3]
//       CHECK:   %[[INPUT_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[TILED_OUTD0:.+]] = affine.apply #[[MAP0]]()[%[[OUTD0]]]
//   CHECK-DAG:   %[[TILED_OUTD1:.+]] = affine.apply #[[MAP1]]()[%[[OUTD1]]]
//   CHECK-DAG:   %[[OUTPUT_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
//  CHECK-SAME:       !flow.dispatch.tensor<writeonly:tensor<?x?x8x4xf32>>{%[[TILED_OUTD0]], %[[TILED_OUTD1]]}
//       CHECK:   %[[INPUT:.+]] = flow.dispatch.tensor.load %[[INPUT_BINDING]]
//       CHECK:   %[[EMPTY:.+]] = tensor.empty
//       CHECK:   %[[PACK:.+]] = tensor.pack
//  CHECK-SAME:       %[[INPUT]] padding_value(%[[CST]] : f32)
//  CHECK-SAME:       inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %[[EMPTY]]
//       CHECK:   flow.dispatch.tensor.store %[[PACK]], %[[OUTPUT_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_OUTD0]], %[[TILED_OUTD1]], 8, 4], strides = [1, 1, 1, 1]

// -----

func.func @unset_encoding_dynamic() attributes {
   hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %d0 = hal.interface.constant.load [0] : index
  %d1 = hal.interface.constant.load [1] : index
  %outd0 = hal.interface.constant.load [2] : index
  %outd1 = hal.interface.constant.load [3] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>>{%d0, %d1}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%outd0, %outd1}
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>>{%d0, %d1}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>
  %3 = iree_linalg_ext.unset_encoding %2
      : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>> -> tensor<?x?xf32>
  %4 = tensor.extract_slice %3[0, 0] [%outd0, %outd1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [%outd0, %outd1], strides = [1, 1]
      : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%outd0, %outd1}
  return
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//       CHECK: func @unset_encoding_dynamic()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[D0:.+]] = hal.interface.constant.load[0]
//   CHECK-DAG:   %[[D1:.+]] = hal.interface.constant.load[1]
//   CHECK-DAG:   %[[OUTD0:.+]] = hal.interface.constant.load[2]
//   CHECK-DAG:   %[[OUTD1:.+]] = hal.interface.constant.load[3]
//   CHECK-DAG:   %[[TILED_D0:.+]] = affine.apply #[[MAP0]]()[%[[D0]]]
//   CHECK-DAG:   %[[TILED_D1:.+]] = affine.apply #[[MAP1]]()[%[[D1]]]
//   CHECK-DAG:   %[[INPUT_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x4xf32>>{%[[TILED_D0]], %[[TILED_D1]]}
//   CHECK-DAG:   %[[OUTPUT_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
//       CHECK:   %[[INPUT:.+]] = flow.dispatch.tensor.load %[[INPUT_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_D0]], %[[TILED_D1]], 8, 4], strides = [1, 1, 1, 1]
//       CHECK:   %[[EMPTY:.+]] = tensor.empty(%[[OUTD0]], %[[OUTD1]])
//       CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[INPUT]]
//  CHECK-SAME:       inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %[[EMPTY]]
//   CHECK-DAG:   flow.dispatch.tensor.store %[[UNPACK]], %[[OUTPUT_BINDING]]

// -----

func.func @matmul_lowering_f32f32f32_generic() attributes {
   hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f32, f32, f32]>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>>{%M, %K}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f32, f32, f32]>>>{%K, %N}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f32, f32, f32]>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>>>{%M, %N}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, element_types = [f32, f32, f32]>>,
                   tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RHS, element_types = [f32, f32, f32]>>)
      outs(%5 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>>)
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = RESULT, element_types = [f32, f32, f32]>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//      CHECK: func @matmul_lowering_f32f32f32_generic()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//  CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[MAP1]]()[%[[K]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x4xf32>>{%[[TILED_M]], %[[TILED_K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x4xf32>>{%[[TILED_N]], %[[TILED_K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xf32>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 8, 4], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 8, 4], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
