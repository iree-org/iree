// RUN: iree-opt --split-input-file --iree-stream-encode-host-tensors %s | FileCheck %s

// CHECK-LABEL: @tensorSizeOfUnalignedPackedI1
util.func public @tensorSizeOfUnalignedPackedI1() -> index {
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
  %0 = stream.tensor.sizeof tensor<12xi1, #iree_encoding.packed_storage> : index
  // CHECK: return %[[C2]] : index
  util.return %0 : index
}

// -----

// CHECK-LABEL: @tensorSizeOfAlignedPackedI1
util.func public @tensorSizeOfAlignedPackedI1() -> index {
  // CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
  %0 = stream.tensor.sizeof tensor<24xi1, #iree_encoding.packed_storage> : index
  // CHECK: util.return %[[C3]] : index
  util.return %0 : index
}

// -----

#encoding_layout = #iree_cpu.vmvx_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [4, 16], outerDimsPerm = [0, 1]}}>
#encoding = #iree_encoding.layout<[#encoding_layout]>
util.func public @sizeof_lhs_encoding_dynamic_using_layouts(%arg0: index, %arg1: index) -> index {
  %0 = stream.tensor.sizeof tensor<?x?xf32, #encoding>{%arg0, %arg1} : index
  util.return %0 : index
}
// CHECK-LABEL: @sizeof_lhs_encoding_dynamic_using_layouts
// CHECK-DAG:     %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK:         %[[CEIL_DIV_D0:.+]] = arith.ceildivsi %arg0, %[[C4]]
// CHECK:         %[[PAD_D0:.+]] = arith.muli %[[CEIL_DIV_D0]], %[[C4]]
// CHECK:         %[[CEIL_DIV_D1:.+]] = arith.ceildivsi %arg1, %[[C16]]
// CHECK:         %[[PAD_D1:.+]] = arith.muli %[[CEIL_DIV_D1]], %[[C16]]
// CHECK:         %[[T0:.+]] = arith.muli %[[PAD_D0]], %[[C4]]
// CHECK:         %[[T1:.+]] = arith.muli %[[T0]], %[[PAD_D1]]
// CHECK:         return %[[T1]]

// -----

#encoding_layout = #iree_cpu.vmvx_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [4, 16], outerDimsPerm = [0, 1]}}>
#encoding = #iree_encoding.layout<[#encoding_layout]>
util.func public @sizeof_lhs_encoding_partially_dynamic_using_layouts(%arg0: index) -> index {
  %0 = stream.tensor.sizeof tensor<10x?xf32, #encoding>{%arg0} : index
  util.return %0 : index
}
// CHECK-LABEL: @sizeof_lhs_encoding_partially_dynamic_using_layouts
// CHECK-DAG:     %[[C48:.+]] = arith.constant 48 : index
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK:         %[[CEIL_DIV_D1:.+]] = arith.ceildivsi %arg0, %[[C16]]
// CHECK:         %[[PAD_D1:.+]] = arith.muli %[[CEIL_DIV_D1]], %[[C16]]
// CHECK:         %[[T0:.+]] = arith.muli %[[PAD_D1]], %[[C48]]
// CHECK:         return %[[T0]]

// -----

// In GEMM, the RHS has the `(M, N, K) -> (K, N)` layout. The  tile sizes
// (i.e., [8, 16]) are for [dim_1, dim_0] in the encoding_info, where dim_1 is
// N-dimension and dim_0 is K-dimension.
#encoding_layout = #iree_cpu.vmvx_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [1, 0], innerTileSizes = [8, 16], outerDimsPerm = [1, 0]}}>
#encoding = #iree_encoding.layout<[#encoding_layout]>
util.func public @sizeof_rhs_encoding_dynamic_using_layouts(%arg0: index, %arg1: index) -> index {
  %0 = stream.tensor.sizeof tensor<?x?xf32, #encoding>{%arg0, %arg1} : index
  util.return %0 : index
}
// CHECK-LABEL: @sizeof_rhs_encoding_dynamic_using_layouts
// CHECK-DAG:     %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:     %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK:         %[[CEIL_DIV_D1:.+]] = arith.ceildivsi %arg1, %[[C8]]
// CHECK:         %[[PAD_D1:.+]] = arith.muli %[[CEIL_DIV_D1]], %[[C8]]
// CHECK:         %[[CEIL_DIV_D0:.+]] = arith.ceildivsi %arg0, %[[C16]]
// CHECK:         %[[PAD_D0:.+]] = arith.muli %[[CEIL_DIV_D0]], %[[C16]]
// CHECK:         %[[T0:.+]] = arith.muli %[[PAD_D0]], %[[C4]]
// CHECK:         %[[T1:.+]] = arith.muli %[[T0]], %[[PAD_D1]]
// CHECK:         return %[[T1]]

// -----

#encoding_layout = #iree_cpu.vmvx_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [4, 8], outerDimsPerm = [0, 1]}}>
#encoding = #iree_encoding.layout<[#encoding_layout]>
util.func public @sizeof_result_encoding_dynamic_using_layouts(%arg0: index, %arg1: index) -> index {
  %0 = stream.tensor.sizeof tensor<?x?xf32, #encoding>{%arg0, %arg1} : index
  util.return %0 : index
}
// CHECK-LABEL: @sizeof_result_encoding_dynamic_using_layouts
// CHECK-DAG:     %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:     %[[C8:.+]] = arith.constant 8 : index
// CHECK:         %[[CEIL_DIV_D0:.+]] = arith.ceildivsi %arg0, %[[C4]]
// CHECK:         %[[PAD_D0:.+]] = arith.muli %[[CEIL_DIV_D0]], %[[C4]]
// CHECK:         %[[CEIL_DIV_D1:.+]] = arith.ceildivsi %arg1, %[[C8]]
// CHECK:         %[[PAD_D1:.+]] = arith.muli %[[CEIL_DIV_D1]], %[[C8]]
// CHECK:         %[[T0:.+]] = arith.muli %[[PAD_D0]], %[[C4]]
// CHECK:         %[[T1:.+]] = arith.muli %[[T0]], %[[PAD_D1]]
// CHECK:         return %[[T1]]

// -----

// The M-dimension inner tile is not present because it broadcasts across the
// M-dimension. We do not need to pack the M-dimension in this case.
#encoding_layout = #iree_cpu.vmvx_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [1], innerTileSizes = [16], outerDimsPerm = [0, 1]}}>
#encoding = #iree_encoding.layout<[#encoding_layout]>
util.func public @sizeof_lhs_encoding_with_bcast_across_m_dim_dynamic_using_layouts(%arg0: index, %arg1: index) -> index {
  %0 = stream.tensor.sizeof tensor<?x?xf32, #encoding>{%arg0, %arg1} : index
  util.return %0 : index
}
// CHECK-LABEL: @sizeof_lhs_encoding_with_bcast_across_m_dim_dynamic_using_layouts
// CHECK-DAG:     %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK:         %[[CEIL_DIV_D1:.+]] = arith.ceildivsi %arg1, %[[C16]]
// CHECK:         %[[PAD_D1:.+]] = arith.muli %[[CEIL_DIV_D1]], %[[C16]]
//
// Multiplied by 4 because f32 has 4 bytes.
//
// CHECK:         %[[T0:.+]] = arith.muli %arg0, %[[C4]]
// CHECK:         %[[T1:.+]] = arith.muli %[[T0]], %[[PAD_D1]]
// CHECK:         return %[[T1]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#no_pad_layout = #iree_encoding.padding<[0, 0]>
#no_pad_encoding = #iree_encoding.layout<[#no_pad_layout]>
#pad_layout_a = #iree_encoding.padding<[0, 64]>
#pad_encoding_a = #iree_encoding.layout<[#pad_layout_a]>
#pad_layout_b = #iree_encoding.padding<[64, 0]>
#pad_encoding_b = #iree_encoding.layout<[#pad_layout_b]>
util.func public @sizeof_lhs_pad_encoding_static() -> index, index, index {
  %0 = stream.tensor.sizeof tensor<2048x4096xf16, #no_pad_encoding>{} : index
  %1 = stream.tensor.sizeof tensor<2048x4096xf16, #pad_encoding_a>{} : index
  %2 = stream.tensor.sizeof tensor<2048x4096xf16, #pad_encoding_b>{} : index
  util.return %0, %1, %2 : index, index, index
}

// We expect (2048 + pad[0]) * (4096 + pad[1]) * (16 / 8).

// CHECK-LABEL: @sizeof_lhs_pad_encoding_static
// CHECK-DAG:     %[[CST_A:.+]] = arith.constant 16777216 : index
// CHECK-DAG:     %[[CST_B:.+]] = arith.constant 17039360 : index
// CHECK-DAG:     %[[CST_C:.+]] = arith.constant 17301504 : index
// CHECK:         return %[[CST_A]], %[[CST_B]], %[[CST_C]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#no_pad_layout = #iree_encoding.padding<[0, 0]>
#no_pad_encoding = #iree_encoding.layout<[#no_pad_layout]>
#pad_layout_a = #iree_encoding.padding<[0, 64]>
#pad_encoding_a = #iree_encoding.layout<[#pad_layout_a]>
#pad_layout_b = #iree_encoding.padding<[64, 0]>
#pad_encoding_b = #iree_encoding.layout<[#pad_layout_b]>
util.func public @sizeof_rhs_pad_encoding_dynamic(%arg0 : index, %arg1 : index) -> index, index, index, index {
  %0 = stream.tensor.sizeof tensor<2048x?xf16, #no_pad_encoding>{%arg0} : index
  %1 = stream.tensor.sizeof tensor<?x4096xf16, #pad_encoding_a>{%arg0} : index
  %2 = stream.tensor.sizeof tensor<?x4096xf16, #pad_encoding_b>{%arg0} : index
  %3 = stream.tensor.sizeof tensor<?x?xf16, #pad_encoding_b>{%arg0, %arg1} : index
  util.return %0, %1, %2, %3 : index, index, index, index
}

// CHECK-LABEL: @sizeof_rhs_pad_encoding_dynamic
// CHECK-DAG:     %[[CST_2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[CST_64:.+]] = arith.constant 64 : index
// CHECK-DAG:     %[[CST_4096:.+]] = arith.constant 4096 : index
// CHECK-DAG:     %[[CST_8192:.+]] = arith.constant 8192 : index
// CHECK-DAG:     %[[CST_8320:.+]] = arith.constant 8320 : index
// CHECK:         %[[A:.+]] = arith.muli %arg0, %[[CST_4096]] overflow<nsw>
// CHECK:         %[[B:.+]] = arith.muli %arg0, %[[CST_8320]] overflow<nsw>
// CHECK:         %[[C_0:.+]] = arith.addi %arg0, %[[CST_64]] overflow<nsw>
// CHECK:         %[[C_1:.+]] = arith.muli %[[C_0]], %[[CST_8192]] overflow<nsw>
// CHECK:         %[[D_0:.+]] = arith.addi %arg0, %[[CST_64]] overflow<nsw>
// CHECK:         %[[D_1:.+]] = arith.muli %[[D_0]], %arg1 overflow<nsw>
// CHECK:         %[[D_2:.+]] = arith.muli %[[D_1]], %[[CST_2]] overflow<nsw>
// CHECK:         return %[[A]], %[[B]], %[[C_1]], %[[D_2]]

// -----

#encoding_layout_0 = #iree_cpu.cpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [4, 8], outerDimsPerm = [0, 1]}}>
#encoding_layout_1 = #iree_cpu.vmvx_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [2, 16], outerDimsPerm = [0, 1]}}>
#encoding_layout_2 = #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [128, 16], outerDimsPerm = [0, 1]}}>
#encoding = #iree_encoding.layout<[#encoding_layout_0, #encoding_layout_1, #encoding_layout_2]>
util.func public @sizeof_multi_encoding_layouts(%arg0: index, %arg1: index) -> index {
  %0 = stream.tensor.sizeof tensor<?x?xf32, #encoding>{%arg0, %arg1} : index
  util.return %0 : index
}
// CHECK-LABEL: @sizeof_multi_encoding_layouts
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:     %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:     %[[C128:.+]] = arith.constant 128 : index
//
// Check for the first layout.
//
// CHECK-DAG:     %[[CEIL_DIV_D0:.+]] = arith.ceildivsi %arg0, %[[C4]]
// CHECK-DAG:     %[[PAD_D0:.+]] = arith.muli %[[CEIL_DIV_D0]], %[[C4]]
// CHECK-DAG:     %[[CEIL_DIV_D1:.+]] = arith.ceildivsi %arg1, %[[C8]]
// CHECK-DAG:     %[[PAD_D1:.+]] = arith.muli %[[CEIL_DIV_D1]], %[[C8]]
// CHECK-DAG:     %[[T0:.+]] = arith.muli %[[PAD_D0]], %[[C4]]
// CHECK-DAG:     %[[SIZE0:.+]] = arith.muli %[[T0]], %[[PAD_D1]]
//
// Check for the second layout.
//
// CHECK-DAG:     %[[CEIL_DIV_D0:.+]] = arith.ceildivsi %arg0, %[[C2]]
// CHECK-DAG:     %[[PAD_D0:.+]] = arith.muli %[[CEIL_DIV_D0]], %[[C2]]
// CHECK-DAG:     %[[CEIL_DIV_D1:.+]] = arith.ceildivsi %arg1, %[[C16]]
// CHECK-DAG:     %[[PAD_D1:.+]] = arith.muli %[[CEIL_DIV_D1]], %[[C16]]
// CHECK-DAG:     %[[T1:.+]] = arith.muli %[[PAD_D0]], %[[C4]]
// CHECK-DAG:     %[[SIZE1:.+]] = arith.muli %[[T1]], %[[PAD_D1]]
//
// Check for the third layout.
//
// CHECK-DAG:     %[[CEIL_DIV_D0:.+]] = arith.ceildivsi %arg0, %[[C128]]
// CHECK-DAG:     %[[PAD_D0:.+]] = arith.muli %[[CEIL_DIV_D0]], %[[C128]]
// CHECK-DAG:     %[[CEIL_DIV_D1:.+]] = arith.ceildivsi %arg1, %[[C16]]
// CHECK-DAG:     %[[PAD_D1:.+]] = arith.muli %[[CEIL_DIV_D1]], %[[C16]]
// CHECK-DAG:     %[[T2:.+]] = arith.muli %[[PAD_D0]], %[[C4]]
// CHECK-DAG:     %[[SIZE2:.+]] = arith.muli %[[T2]], %[[PAD_D1]]
//
// Return the max value.
//
// CHECK-DAG:     %[[RES_0_1:.+]] = arith.maxui %[[SIZE0]], %[[SIZE1]]
// CHECK-DAG:     %[[RES:.+]] = arith.maxui %[[RES_0_1]], %[[SIZE2]]
// CHECK:         return %[[RES]]
