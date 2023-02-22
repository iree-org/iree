// RUN: iree-opt --iree-llvmcpu-materialize-encoding --canonicalize --cse --split-input-file %s | FileCheck %s

func.func @set_encoding_op() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %d0 = hal.interface.constant.load [0] : index
  %d1 = hal.interface.constant.load [1] : index
  %outd0 = hal.interface.constant.load [2] : index
  %outd1 = hal.interface.constant.load [3] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%d0, %d1}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>{%outd0, %outd1}
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%d0, %d1} -> tensor<?x?xf32>
  %p0 = affine.apply affine_map<()[s0, s1] -> (-s0 + s1)>()[%d0, %outd0]
  %p1 = affine.apply affine_map<()[s0, s1] -> (-s0 + s1)>()[%d1, %outd1]
  %padded = tensor.pad %2 low[0, 0] high[%p0, %p1] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<?x?xf32>
  %3 = iree_linalg_ext.set_encoding %padded : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [%outd0, %outd1], strides = [1, 1]
      : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
      -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>{%outd0, %outd1}
  return
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//       CHECK: func @set_encoding_op()
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
//       CHECK:   %[[EMPTY:.+]] = tensor.empty(%[[TILED_OUTD0]], %[[TILED_OUTD1]])
//       CHECK:   %[[PACK:.+]] = iree_linalg_ext.pack
//  CHECK-SAME:       %[[INPUT]] padding_value(%[[CST]] : f32)
//  CHECK-SAME:       inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %[[EMPTY]]
//       CHECK:   flow.dispatch.tensor.store %[[PACK]], %[[OUTPUT_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_OUTD0]], %[[TILED_OUTD1]], 8, 4], strides = [1, 1, 1, 1]

// -----

func.func @unset_encoding_op() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %d0 = hal.interface.constant.load [0] : index
  %d1 = hal.interface.constant.load [1] : index
  %outd0 = hal.interface.constant.load [2] : index
  %outd1 = hal.interface.constant.load [3] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>{%d0, %d1}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%outd0, %outd1}
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>{%d0, %d1}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
  %3 = iree_linalg_ext.unset_encoding %2
      : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>> -> tensor<?x?xf32>
  %4 = tensor.extract_slice %3[0, 0] [%outd0, %outd1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [%outd0, %outd1], strides = [1, 1]
      : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%outd0, %outd1}
  return
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//       CHECK: func @unset_encoding_op()
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
//       CHECK:   %[[UNPACK:.+]] = iree_linalg_ext.unpack %[[INPUT]]
//  CHECK-SAME:       inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %[[EMPTY]]
//   CHECK-DAG:   flow.dispatch.tensor.store %[[UNPACK]], %[[OUTPUT_BINDING]]

// -----

func.func @matmul_lowering_f32f32f32_generic() {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>{%M, %K}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>>{%K, %N}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>>{%M, %N}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>,
                   tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>)
      outs(%5 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>)
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>>{%M, %N}
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

// -----

func.func @matmul_lowering_f32f32f32_aarch64() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="aarch64-xyz-xyz"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>{%M, %K}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>>{%K, %N}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>>{%M, %N}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>,
                   tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>)
      outs(%5 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>)
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//      CHECK: func @matmul_lowering_f32f32f32_aarch64()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x1xf32>>{%[[TILED_M]], %[[K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x1xf32>>{%[[TILED_N]], %[[K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xf32>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]

// -----

func.func @matmul_lowering_f32f32f32_x86_64() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>{%M, %K}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>>{%K, %N}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>>{%M, %N}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>,
                   tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>)
      outs(%5 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>)
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//      CHECK: func @matmul_lowering_f32f32f32_x86_64()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x1xf32>>{%[[TILED_M]], %[[K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP1]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x4x1xf32>>{%[[TILED_N]], %[[K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x4xf32>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 4, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 4], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 4], strides = [1, 1, 1, 1]

// -----

func.func @matmul_lowering_f32f32f32_x86_64_avx2() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx2"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>{%M, %K}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>>{%K, %N}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>>{%M, %N}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>,
                   tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>)
      outs(%5 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>)
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//      CHECK: func @matmul_lowering_f32f32f32_x86_64_avx2()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x1xf32>>{%[[TILED_M]], %[[K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x1xf32>>{%[[TILED_N]], %[[K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xf32>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]

// -----

func.func @matmul_lowering_f32f32f32_x86_64_avx512f() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>{%M, %K}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>>{%K, %N}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>>{%M, %N}
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>,
                   tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>)
      outs(%5 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>)
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//      CHECK: func @matmul_lowering_f32f32f32_x86_64_avx512f()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x1xf32>>{%[[TILED_M]], %[[K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x1xf32>>{%[[TILED_N]], %[[K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x16x16xf32>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 16, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 16, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]

// -----

func.func @matmul_lowering_i8i8i32_aarch64() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="aarch64-xyz-xyz"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_LHS>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RHS>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_LHS>>>{%M, %K}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_LHS>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RHS>>>{%K, %N}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RHS>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>>{%M, %N}
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_LHS>>,
                   tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RHS>>)
      outs(%5 : tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>)
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//      CHECK: func @matmul_lowering_i8i8i32_aarch64()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x1xi8>>{%[[TILED_M]], %[[K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x1xi8>>{%[[TILED_N]], %[[K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xi32>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]

// -----

func.func @matmul_lowering_i8i8i32_aarch64_dotprod() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+dotprod"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_LHS>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RHS>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_LHS>>>{%M, %K}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_LHS>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RHS>>>{%K, %N}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RHS>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>>{%M, %N}
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_LHS>>,
                   tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RHS>>)
      outs(%5 : tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>)
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//      CHECK: func @matmul_lowering_i8i8i32_aarch64_dotprod()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//  CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[MAP1]]()[%[[K]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x4xi8>>{%[[TILED_M]], %[[TILED_K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x4xi8>>{%[[TILED_N]], %[[TILED_K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xi32>>{%[[TILED_M]], %[[TILED_N]]}
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

// -----

func.func @matmul_lowering_i8i8i32_aarch64_i8mm() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+dotprod,+i8mm"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_LHS>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RHS>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_LHS>>>{%M, %K}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_LHS>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RHS>>>{%K, %N}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RHS>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>>{%M, %N}
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_LHS>>,
                   tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RHS>>)
      outs(%5 : tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>)
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//      CHECK: func @matmul_lowering_i8i8i32_aarch64_i8mm()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//  CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[MAP0]]()[%[[K]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x8xi8>>{%[[TILED_M]], %[[TILED_K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x8xi8>>{%[[TILED_N]], %[[TILED_K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xi32>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 8, 8], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 8, 8], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]

// -----

func.func @matmul_lowering_i8i8i32_x86_64_avx2() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx2"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_LHS>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RHS>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_LHS>>>{%M, %K}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_LHS>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RHS>>>{%K, %N}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RHS>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>>{%M, %N}
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_LHS>>,
                   tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RHS>>)
      outs(%5 : tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>)
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
//      CHECK: func @matmul_lowering_i8i8i32_x86_64_avx2()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//  CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[MAP1]]()[%[[K]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x2xi8>>{%[[TILED_M]], %[[TILED_K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x2xi8>>{%[[TILED_N]], %[[TILED_K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xi32>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 8, 2], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 8, 2], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]

// -----

func.func @matmul_lowering_i8i8i32_x86_64_avx512bw() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512bw"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_LHS>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RHS>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_LHS>>>{%M, %K}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_LHS>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RHS>>>{%K, %N}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RHS>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>>{%M, %N}
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_LHS>>,
                   tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RHS>>)
      outs(%5 : tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>)
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
//      CHECK: func @matmul_lowering_i8i8i32_x86_64_avx512bw()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//  CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[MAP1]]()[%[[K]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x2xi8>>{%[[TILED_M]], %[[TILED_K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x2xi8>>{%[[TILED_N]], %[[TILED_K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x16x16xi32>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 16, 2], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 16, 2], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]

// -----

func.func @matmul_lowering_i8i8i32_x86_64_avx512vnni() attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load[0] : index
  %N = hal.interface.constant.load[1] : index
  %K = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_LHS>>>{%M, %K}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RHS>>>{%K, %N}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_LHS>>>{%M, %K}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_LHS>>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RHS>>>{%K, %N}
      -> tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RHS>>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>>{%M, %N}
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_LHS>>,
                   tensor<?x?xi8, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RHS>>)
      outs(%5 : tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>)
      -> tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #iree_linalg_ext.encoding<MATMUL_I8I8I32_RESULT>>>{%M, %N}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//      CHECK: func @matmul_lowering_i8i8i32_x86_64_avx512vnni()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//  CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[MAP1]]()[%[[K]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x4xi8>>{%[[TILED_M]], %[[TILED_K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x16x4xi8>>{%[[TILED_N]], %[[TILED_K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x16x16xi32>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 16, 4], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 16, 4], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 16, 16], strides = [1, 1, 1, 1]
