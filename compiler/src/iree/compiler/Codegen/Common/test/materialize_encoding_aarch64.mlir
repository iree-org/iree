// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding))" --split-input-file %s | FileCheck %s --check-prefixes=CHECK,NO-SVE
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding))" --iree-llvmcpu-enable-scalable-vectorization=true --split-input-file %s | FileCheck %s --check-prefixes=CHECK,WITH-SVE

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [bf16, bf16, bf16], user_indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>], iteration_sizes = [?, ?, ?]>
func.func @matmul_LHS() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+sve", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
}{
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x16xbf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x16xbf16, #encoding>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 16], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x16xbf16>> -> tensor<8x16xbf16>
  %3 = iree_encoding.set_encoding %2 : tensor<8x16xbf16> -> tensor<8x16xbf16, #encoding>
  iree_tensor_ext.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [8, 16], strides = [1, 1] : tensor<8x16xbf16, #encoding> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x16xbf16,  #encoding>>
  return
}
/// NOTE: No scalable tiles for LHS, hence no difference between NO-SVE and WITH-SVE

// CHECK-LABEL: func.func @matmul_LHS
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[IN_BINDING:.+]] = hal.interface.binding.subspan {{.+}} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x16xbf16>>
// CHECK-DAG:     %[[OUT_BINDING:.+]] = hal.interface.binding.subspan {{.+}} : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x16x8x1xbf16>>
// CHECK:         %[[SRC:.+]] = iree_tensor_ext.dispatch.tensor.load %[[IN_BINDING]]
// CHECK-DAG:     %[[INIT:.+]] = tensor.empty() : tensor<1x16x8x1xbf16>
// CHECK:         %[[PACK:.+]] = linalg.pack %[[SRC]]
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [8, 1]
// CHECK-SAME:      into %[[INIT]] : tensor<8x16xbf16> -> tensor<1x16x8x1xbf16>
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[PACK]], %[[OUT_BINDING]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#encoding = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [bf16, bf16, bf16], user_indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>], iteration_sizes = [?, ?, ?]>
func.func @matmul_RHS() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+sve", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
}{
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x16xbf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x16xbf16, #encoding>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 16], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x16xbf16>> -> tensor<8x16xbf16>
  %3 = iree_encoding.set_encoding %2 : tensor<8x16xbf16> -> tensor<8x16xbf16, #encoding>
  iree_tensor_ext.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [8, 16], strides = [1, 1] : tensor<8x16xbf16, #encoding> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x16xbf16,  #encoding>>
  return
}
/// NOTE: For RHS, the inner tile correspondong to the "N" dimension is
/// scalable, hence NO-SVE and WITH-SVE differ!

// WITH-SVE: #[[$MAP:.+]] = affine_map<()[s0] -> (16 ceildiv s0)>

// CHECK-LABEL: func.func @matmul_RHS
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// WITH-SVE-DAG:  %[[C8:.*]] = arith.constant 8 : index
// WITH-SVE-DAG:  %[[PAD:.+]] = arith.constant 0.000000e+00 : bf16

/// Input binding
// CHECK-DAG:     %[[IN_BINDING:.+]] = hal.interface.binding.subspan {{.+}} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x16xbf16>>

/// SVE: the number of outer tiles corresponding to the inner scalable tile
// WITH-SVE:      %[[VSCALE:.*]] = vector.vscale
// WITH-SVE:      %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C8]] : index
// WITH-SVE:      %[[OUTER_DIM:.*]] = affine.apply #[[$MAP]]()[%[[C8_VSCALE]]]

/// Output binding
// NO-SVE-DAG:     %[[OUT_BINDING:.+]] = hal.interface.binding.subspan {{.+}} : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x8x8x1xbf16>>
// WITH-SVE-DAG:   %[[OUT_BINDING:.+]] = hal.interface.binding.subspan {{.+}} : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x8x?x1xbf16>>

/// Input
// CHECK:         %[[SRC:.+]] = iree_tensor_ext.dispatch.tensor.load %[[IN_BINDING]]

/// Init the output tensor
// NO-SVE-DAG:     %[[INIT:.+]] = tensor.empty() : tensor<2x8x8x1xbf16>
// WITH-SVE-DAG:   %[[INIT:.*]] = tensor.empty(%[[OUTER_DIM]], %[[C8_VSCALE]]) : tensor<?x8x?x1xbf16>

/// The newly materialised Pack Op (SVE includes padding)
// CHECK:         %[[PACK:.+]] = linalg.pack %[[SRC]]
// WITH-SVE-SAME:    padding_value(%[[PAD]] : bf16)
// NO-SVE-NOT:        padding_value

// CHECK-SAME:      outer_dims_perm = [1, 0]
// CHECK-SAME:      inner_dims_pos = [1, 0]

// NO-SVE-SAME:      inner_tiles = [8, 1]
// NO-SVE-SAME:      into %[[INIT]] : tensor<8x16xbf16> -> tensor<2x8x8x1xbf16>

// WITH-SVE-SAME:    inner_tiles = [%[[C8_VSCALE]], 1]
// WITH-SVE-SAME:    into %[[INIT]] : tensor<8x16xbf16> -> tensor<?x8x?x1xbf16>

// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[PACK]], %[[OUT_BINDING]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(b, m, n, k) -> (b, m, k)>
#map1 = affine_map<(b, m, n, k) -> (b, k, n)>
#map2 = affine_map<(b, m, n, k) -> (b, m, n)>
#encoding = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [128, 32, 320, ?]>
func.func @batch_matmul_RHS() attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+sve", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = arith.index_castui %0 {stream.alignment = 64 : index} : i32 to index
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x32x320xf32>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x32x320xf32, #encoding>>
  %4 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [128, 32, 320], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x32x320xf32>> -> tensor<128x32x320xf32>
  %5 = iree_encoding.set_encoding %4 : tensor<128x32x320xf32> -> tensor<128x32x320xf32, #encoding>
  iree_tensor_ext.dispatch.tensor.store %5, %3, offsets = [0, 0, 0], sizes = [128, 32, 320], strides = [1, 1, 1]
    : tensor<128x32x320xf32, #encoding>
    -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x32x320xf32, #encoding>>
  return
}
/// NOTE: For RHS, the inner tile correspondong to the "N" dimension is
/// scalable, hence NO-SVE and WITH-SVE differ!

// WITH-SVE: #[[$MAP:.+]] = affine_map<()[s0] -> (320 ceildiv s0)>

// CHECK-LABEL:   func.func @batch_matmul_RHS()
// WITH-SVE-DAG:    %[[PAD:.+]] = arith.constant 0.000000e+00 : f32
// WITH-SVE-DAG:    %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:       %[[C0:.+]] = arith.constant 0 : index

/// Input binding
// CHECK:           %[[IN_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(0) {{.+}} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x32x320xf32>>

/// SVE: the number of outer tiles corresponding to the inner scalable tile
// WITH-SVE:        %[[VSCALE:.+]] = vector.vscale
// WITH-SVE:        %[[C8_VSCALE:.+]] = arith.muli %[[VSCALE]], %[[C8]] : index
// WITH-SVE:        %[[OUTER_DIM:.+]] = affine.apply #[[$MAP]]()[%[[C8_VSCALE]]]

/// Output binding
// WITH-SVE:        %[[OUT_BINDING:.+]] = hal.interface.binding.subspan {{.+}} : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x?x32x?x1xf32>>
// NO-SVE:          %[[OUT_BINDING:.+]] = hal.interface.binding.subspan {{.+}} : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x40x32x8x1xf32>>

/// Input
// CHECK:           %[[SRC:.+]] = iree_tensor_ext.dispatch.tensor.load %[[IN_BINDING]]

/// Init the output tensor
// NO-SVE:          %[[INIT:.+]] = tensor.empty() : tensor<128x40x32x8x1xf32>
// WITH-SVE:        %[[INIT:.+]] = tensor.empty(%[[OUTER_DIM]], %[[C8_VSCALE]]) : tensor<128x?x32x?x1xf32>

/// The newly materialised Pack Op (SVE includes padding!)
// CHECK:           %[[PACK:.+]] = linalg.pack %[[SRC]]
// WITH-SVE-SAME:     padding_value(%[[PAD]] : f32)
// NO-SVE-NOT:        padding_value

// CHECK-SAME:        outer_dims_perm = [0, 2, 1]
// CHECK-SAME:        inner_dims_pos = [2, 1]

// NO-SVE-SAME:       inner_tiles = [8, 1] into %[[INIT]] : tensor<128x32x320xf32> -> tensor<128x40x32x8x1xf32>

// WITH-SVE-SAME:     inner_tiles = [%[[C8_VSCALE]], 1] into %[[INIT]] : tensor<128x32x320xf32> -> tensor<128x?x32x?x1xf32>

// CHECK:           iree_tensor_ext.dispatch.tensor.store %[[PACK]], %[[OUT_BINDING]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(b, m, n, k) -> (b, m, k)>
#map1 = affine_map<(b, m, n, k) -> (b, k, n)>
#map2 = affine_map<(b, m, n, k) -> (b, m, n)>
#encoding = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [128, 80, 320, ?]>
func.func @batch_matmul_RETURN_unset() attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+sve", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = arith.index_castui %0 {stream.alignment = 64 : index} : i32 to index
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x80x320xf32, #encoding>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x80x320xf32>>
  %4 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [128, 80, 320], strides = [1, 1, 1]
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x80x320xf32, #encoding>>
    -> tensor<128x80x320xf32, #encoding>
  %5 = iree_encoding.unset_encoding %4 : tensor<128x80x320xf32, #encoding> -> tensor<128x80x320xf32>
  iree_tensor_ext.dispatch.tensor.store %5, %3, offsets = [0, 0, 0], sizes = [128, 80, 320], strides = [1, 1, 1]
    : tensor<128x80x320xf32>
    -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x80x320xf32>>
  return
}

/// NOTE: For RETURN, the inner tile correspondong to the "N" dimension is
/// scalable, hence NO-SVE and WITH-SVE differ!

// CHECK-LABEL: func @batch_matmul_RETURN_unset
// WITH-SVE-DAG:  %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index

/// SVE: the number of outer tiles corresponding to the inner scalable tile
// WITH-SVE:      %[[VSCALE:.+]] = vector.vscale
// WITH-SVE:      %[[C8_VSCALE:.+]] = arith.muli %[[VSCALE]], %[[C8]] : index
// WITH-SVE:      %[[OUTER_DIM:.+]] = affine.apply #[[$MAP]]()[%[[C8_VSCALE]]]

/// The in and out bindings
// NO-SVE:        %[[IN_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(0) {{.+}} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x10x40x8x8xf32>>
// WITH-SVE:      %[[IN_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(0) {{.+}} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x10x?x8x?xf32>>
// CHECK:         %[[OUT_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(1) {{.+}} : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x80x320xf32>>

// CHECK:         %[[INPUT:.+]] = iree_tensor_ext.dispatch.tensor.load %[[IN_BINDING]]
// CHECK:         %[[EMPTY:.+]] = tensor.empty()

/// The newly materialised UnPack Op
// CHECK:         %[[UNPACK:.+]] = linalg.unpack %[[INPUT]]
// NO-SVE-SAME:       outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, 8] into %[[EMPTY]]
// WITH-SVE-SAME:     outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [8, %[[C8_VSCALE]]] into %[[EMPTY]]

// CHECK:        iree_tensor_ext.dispatch.tensor.store %[[UNPACK]], %[[OUT_BINDING]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iteration_sizes = [16, 1, 16]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iteration_sizes = [16, 1, 16]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iteration_sizes = [16, 1, 16]>
func.func @matvec_shaped_matmul_lowering_f32f32f32_aarch64(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %0 = hal.tensor.import %arg0 "input0" : !hal.buffer_view -> tensor<16x16xf32>
  %1 = hal.tensor.import %arg1 "input1" : !hal.buffer_view -> tensor<16x1xf32>
  %2 = hal.tensor.import %arg2 "input2" : !hal.buffer_view -> tensor<16x1xf32>
  %3 = iree_encoding.set_encoding %0 : tensor<16x16xf32> -> tensor<16x16xf32, #encoding_lhs>
  %4 = iree_encoding.set_encoding %1 : tensor<16x1xf32> -> tensor<16x1xf32, #encoding_rhs>
  %5 = iree_encoding.set_encoding %2 : tensor<16x1xf32> -> tensor<16x1xf32, #encoding_result>
  %6 = linalg.matmul ins(%3, %4 : tensor<16x16xf32, #encoding_lhs>, tensor<16x1xf32, #encoding_rhs>) outs(%5 : tensor<16x1xf32, #encoding_result>) -> tensor<16x1xf32, #encoding_result>
  %7 = iree_encoding.unset_encoding %6 : tensor<16x1xf32, #encoding_result> -> tensor<16x1xf32>
  %8 = hal.tensor.export %7 "output0" : tensor<16x1xf32> -> !hal.buffer_view
  func.return %8 : !hal.buffer_view
}
// CHECK-LABEL: func @matvec_shaped_matmul_lowering_f32f32f32_aarch64(
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins({{.*}} : tensor<1x16x1x1xf32>, tensor<2x16x8x1xf32>)
//  CHECK-SAME:        outs({{.*}} : tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_f32f32f32_aarch64() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {cpu_features="+sve", target_triple="aarch64-xyz-xyz", ukernels = "all", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xf32, #encoding_lhs>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xf32, #encoding_rhs>
  %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
      -> tensor<?x?xf32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf32, #encoding_lhs>,
                   tensor<?x?xf32, #encoding_rhs>)
      outs(%5 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf32, #encoding_result>
      -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
  return
}
//     CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//  WITH-SVE-DAG: #[[$MAP1:.+]] = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
//   CHECK-LABEL: func @matmul_lowering_f32f32f32_aarch64()
//     CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  WITH-SVE-DAG:   %[[C8:.+]] = arith.constant 8 : index
//     CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//     CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//     CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//     CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//         CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//    CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x1xf32>>{%[[TILED_M]], %[[K]]}
///
/// Compute "tiled" N:
///  *  N / 8 for fixed-width tiling,
///  *  N / ( 8 * vscale) for scalable tiling
///
//        NO-SVE:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//      WITH-SVE:   %[[VSCALE:.+]] = vector.vscale
//      WITH-SVE:   %[[C8_VSCALE:.+]] = arith.muli %[[VSCALE]], %[[C8]]
//      WITH-SVE:   %[[TILED_N:.+]] = affine.apply #[[$MAP1]]()[%[[N]], %[[C8_VSCALE]]]
//         CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//   NO-SVE-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x1xf32>>{%[[TILED_N]], %[[K]]}
// WITH-SVE-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?x1xf32>>{%[[TILED_N]], %[[K]], %[[C8_VSCALE]]}
//         CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//   NO-SVE-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x8x8xf32>>{%[[TILED_M]], %[[TILED_N]]}
// WITH-SVE-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x8x?xf32>>{%[[TILED_M]], %[[TILED_N]], %[[C8_VSCALE]]}
//         CHECK:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]
//    CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//         CHECK:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]
//   NO-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
// WITH-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], %[[C8_VSCALE]], 1], strides = [1, 1, 1, 1]
//         CHECK:   %[[OUTS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUTS_BINDING]]
//   NO-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
// WITH-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, %[[C8_VSCALE]]], strides = [1, 1, 1, 1]
//         CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//    CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//    CHECK-SAME:       outs(%[[OUTS]] :
//         CHECK:   iree_tensor_ext.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//   NO-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
// WITH-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, %[[C8_VSCALE]]], strides = [1, 1, 1, 1]
// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>], iteration_sizes = [16, 16]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>], iteration_sizes= [16, 16]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>], iteration_sizes = [16, 16]>
func.func @matvec_lowering_f32f32f32_aarch64(%arg0: tensor<16x16xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>) -> tensor<16xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %3 = iree_encoding.set_encoding %arg0 : tensor<16x16xf32> -> tensor<16x16xf32, #encoding_lhs>
  %4 = iree_encoding.set_encoding %arg1 : tensor<16xf32> -> tensor<16xf32, #encoding_rhs>
  %5 = iree_encoding.set_encoding %arg2 : tensor<16xf32> -> tensor<16xf32, #encoding_result>
  %6 = linalg.matvec ins(%3, %4 : tensor<16x16xf32, #encoding_lhs>, tensor<16xf32, #encoding_rhs>) outs(%5 : tensor<16xf32, #encoding_result>) -> tensor<16xf32, #encoding_result>
  %7 = iree_encoding.unset_encoding %6 : tensor<16xf32, #encoding_result> -> tensor<16xf32>
  func.return %7 : tensor<16xf32>
}
// CHECK-LABEL: func @matvec_lowering_f32f32f32_aarch64(
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins({{.*}} : tensor<1x16x1x1xf32>, tensor<2x16x8x1xf32>)
//  CHECK-SAME:        outs({{.*}} : tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [16, 1, 16]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [16, 1, 16]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [16, 1, 16]>
func.func @matvec_lowering_f32f32f32_aarch64() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x16xf32, #encoding_lhs>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x1xf32, #encoding_rhs>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x1xf32, #encoding_result>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [16, 16], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x16xf32, #encoding_lhs>>
      -> tensor<16x16xf32, #encoding_lhs>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 1], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x1xf32, #encoding_rhs>>
      -> tensor<16x1xf32, #encoding_rhs>
  %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [16, 1], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x1xf32, #encoding_result>>
      -> tensor<16x1xf32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<16x16xf32, #encoding_lhs>,
                   tensor<16x1xf32, #encoding_rhs>)
      outs(%5 : tensor<16x1xf32, #encoding_result>)
      -> tensor<16x1xf32, #encoding_result>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [16, 1], strides = [1, 1]
      : tensor<16x1xf32, #encoding_result>
      -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x1xf32, #encoding_result>>
  return
}
// CHECK-LABEL: func @matvec_lowering_f32f32f32_aarch64()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x16x8x1xf32>>
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x16x1x1xf32>>
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1x2x1x8xf32>>
//       CHECK:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [2, 16, 8, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [1, 16, 1, 1], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [1, 2, 1, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[RHS]], %[[LHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [1, 2, 1, 8], strides = [1, 1, 1, 1]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_f16f16f16_aarch64() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {cpu_features="+sve", target_triple="aarch64-xyz-xyz", ukernels = "all", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf16, #encoding_result>>{%M, %N}
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xf16, #encoding_lhs>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xf16, #encoding_rhs>
  %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf16, #encoding_result>>{%M, %N}
      -> tensor<?x?xf16, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf16, #encoding_lhs>,
                   tensor<?x?xf16, #encoding_rhs>)
      outs(%5 : tensor<?x?xf16, #encoding_result>)
      -> tensor<?x?xf16, #encoding_result>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf16, #encoding_result>
      -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf16, #encoding_result>>{%M, %N}
  return
}
//     CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//  WITH-SVE-DAG: #[[$MAP1:.+]] = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
//   CHECK-LABEL: func @matmul_lowering_f16f16f16_aarch64()
//     CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  WITH-SVE-DAG:   %[[C8:.+]] = arith.constant 8 : index
//     CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(0)
//     CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(1)
//     CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout(#pipeline_layout) ordinal(2)
//     CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//         CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//    CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x1xf16>>{%[[TILED_M]], %[[K]]}
///
/// Compute "tiled" N for fixed-width and scalable tiling, see @matmul_lowering_f32f32f32_aarch64.
///
//        NO-SVE:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//      WITH-SVE:   %[[VSCALE:.+]] = vector.vscale
//      WITH-SVE:   %[[C8_VSCALE:.+]] = arith.muli %[[VSCALE]], %[[C8]]
//      WITH-SVE:   %[[TILED_N:.+]] = affine.apply #[[$MAP1]]()[%[[N]], %[[C8_VSCALE]]]
//         CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//   NO-SVE-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x1xf16>>{%[[TILED_N]], %[[K]]}
// WITH-SVE-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?x1xf16>>{%[[TILED_N]], %[[K]], %[[C8_VSCALE]]}
//         CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//   NO-SVE-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x8x8xf16>>{%[[TILED_M]], %[[TILED_N]]}
// WITH-SVE-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x8x?xf16>>{%[[TILED_M]], %[[TILED_N]], %[[C8_VSCALE]]}
//         CHECK:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]
//    CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
//         CHECK:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]
//   NO-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], 8, 1], strides = [1, 1, 1, 1]
// WITH-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[K]], %[[C8_VSCALE]], 1], strides = [1, 1, 1, 1]
//         CHECK:   %[[OUTS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUTS_BINDING]]
//   NO-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
// WITH-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, %[[C8_VSCALE]]], strides = [1, 1, 1, 1]
//         CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//    CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//    CHECK-SAME:       outs(%[[OUTS]] :
//         CHECK:   iree_tensor_ext.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//   NO-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
// WITH-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, %[[C8_VSCALE]]], strides = [1, 1, 1, 1]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_f32f16f16_aarch64() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {cpu_features="+sve", target_triple="aarch64-xyz-xyz", ukernels = "all", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf16, #encoding_result>>{%M, %N}
  %lhs_f32 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xf32, #encoding_lhs>
  %rhs = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xf16, #encoding_rhs>
  %dest = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf16, #encoding_result>>{%M, %N}
      -> tensor<?x?xf16, #encoding_result>

  %empty = tensor.empty(%M, %K) : tensor<?x?xf16, #encoding_lhs>
  %lhs_f16 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
     ins(%lhs_f32 : tensor<?x?xf32, #encoding_lhs>)
     outs(%empty : tensor<?x?xf16, #encoding_lhs>) {
  ^bb0(%in: f32, %out: f16):
    %17 = arith.truncf %in : f32 to f16
    linalg.yield %17 : f16
  } -> tensor<?x?xf16, #encoding_lhs>
  %6 = linalg.matmul
      ins(%lhs_f16, %rhs : tensor<?x?xf16, #encoding_lhs>,
                   tensor<?x?xf16, #encoding_rhs>)
      outs(%dest : tensor<?x?xf16, #encoding_result>)
      -> tensor<?x?xf16, #encoding_result>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf16, #encoding_result>
      -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf16, #encoding_result>>{%M, %N}
  return
}
//     CHECK-DAG: #[[$MAP_CEILDIV_8:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//     CHECK-DAG: #[[$MAP_IDENTITY_4D:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//  WITH-SVE-DAG: #[[$MAP_VSCALE:.+]] = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
//   CHECK-LABEL: func.func @matmul_lowering_f32f16f16_aarch64()
//  WITH-SVE-DAG:   %[[C8:.+]] = arith.constant 8 : index
//     CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0) : index
//     CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1) : index
//     CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2) : index
//         CHECK:   %[[M_CEILDIV_8:.+]] = affine.apply #[[$MAP_CEILDIV_8]]()[%[[M]]]
//         CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//    CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x1xf32>>{%[[M_CEILDIV_8]], %[[K]]}
//        NO-SVE:   %[[N_CEILDIV_8:.+]] = affine.apply #[[$MAP_CEILDIV_8]]()[%[[N]]]
///
/// Compute "tiled" N for fixed-width and scalable tiling, see @matmul_lowering_f32f32f32_aarch64.
///
//      WITH-SVE:   %[[VSCALE:.+]] = vector.vscale
//      WITH-SVE:   %[[C8_VSCALE:.+]] = arith.muli %[[VSCALE]], %[[C8]]
//      WITH-SVE:   %[[N_CEILDIV_VSCALE:.+]] = affine.apply #[[$MAP_VSCALE]]()[%[[N]], %[[C8_VSCALE]]]
//         CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//   NO-SVE-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x1xf16>>{%[[N_CEILDIV_8]], %[[K]]}
// WITH-SVE-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?x1xf16>>{%[[N_CEILDIV_VSCALE]], %[[K]], %[[C8_VSCALE]]}
//     CHECK-DAG:   %[[OUT_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//   NO-SVE-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x8x8xf16>>{%[[M_CEILDIV_8]], %[[N_CEILDIV_8]]}
// WITH-SVE-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x8x?xf16>>{%[[M_CEILDIV_8]], %[[N_CEILDIV_VSCALE]], %[[C8_VSCALE]]}
//         CHECK:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]], offsets = [0, 0, 0, 0],
//    CHECK-SAME:       sizes = [%[[M_CEILDIV_8]], %[[K]], 8, 1], {{.*}} -> tensor<?x?x8x1xf32>
//         CHECK:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]], offsets = [0, 0, 0, 0],
//   NO-SVE-SAME:       sizes = [%[[N_CEILDIV_8]], %[[K]], 8, 1], {{.*}} -> tensor<?x?x8x1xf16>
// WITH-SVE-SAME:       sizes = [%[[N_CEILDIV_VSCALE]], %[[K]], %[[C8_VSCALE]], 1], {{.*}} -> tensor<?x?x?x1xf16>
//         CHECK:   %[[OUT:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUT_BINDING]], offsets = [0, 0, 0, 0],
//   NO-SVE-SAME:       sizes = [%[[M_CEILDIV_8]], %[[N_CEILDIV_8]], 8, 8], {{.*}} -> tensor<?x?x8x8xf16>
// WITH-SVE-SAME:       sizes = [%[[M_CEILDIV_8]], %[[N_CEILDIV_VSCALE]], 8, %[[C8_VSCALE]]], {{.*}} -> tensor<?x?x8x?xf16>
//         CHECK:   %[[EMPTY:.+]] = tensor.empty(%[[M_CEILDIV_8]], %[[K]]) : tensor<?x?x8x1xf16>
//         CHECK:   %[[LHS_F16:.+]] = linalg.generic {indexing_maps = [#[[$MAP_IDENTITY_4D]], #[[$MAP_IDENTITY_4D]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[LHS]] : tensor<?x?x8x1xf32>) outs(%[[EMPTY]] : tensor<?x?x8x1xf16>) {
//         CHECK:   %[[MMT4D:.+]] = linalg.mmt4d ins(%[[LHS_F16]], %[[RHS]] :
//   NO-SVE-SAME:       tensor<?x?x8x1xf16>, tensor<?x?x8x1xf16>) outs(%[[OUT]] : tensor<?x?x8x8xf16>)
// WITH-SVE-SAME:       tensor<?x?x8x1xf16>, tensor<?x?x?x1xf16>) outs(%[[OUT]] : tensor<?x?x8x?xf16>)
//         CHECK:   iree_tensor_ext.dispatch.tensor.store %[[MMT4D]], %[[OUT_BINDING]],

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_i8i8i32_aarch64() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xi8, #encoding_lhs>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xi8, #encoding_rhs>
  %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
      -> tensor<?x?xi32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #encoding_lhs>,
                   tensor<?x?xi8, #encoding_rhs>)
      outs(%5 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #encoding_result>
      -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  return
}
// CHECK-LABEL: func @matmul_lowering_i8i8i32_aarch64()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8>>{%[[M]], %[[K]]}
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8>>{%[[K]], %[[N]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32>>{%[[M]], %[[N]]}
//       CHECK:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0], sizes = [%[[M]], %[[K]]], strides = [1, 1]
//       CHECK:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0], sizes = [%[[K]], %[[N]]], strides = [1, 1]
//       CHECK:   %[[OUTS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0], sizes = [%[[M]], %[[N]]], strides = [1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.matmul
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0], sizes = [%[[M]], %[[N]]], strides = [1, 1]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_i8i8i32_aarch64_dotprod() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+dotprod,+sve", ukernels = "all", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xi8, #encoding_lhs>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xi8, #encoding_rhs>
  %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
      -> tensor<?x?xi32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #encoding_lhs>,
                   tensor<?x?xi8, #encoding_rhs>)
      outs(%5 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #encoding_result>
      -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  return
}
//     CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//     CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//  WITH-SVE-DAG: #[[$MAP2:.+]] = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
//   CHECK-LABEL: func @matmul_lowering_i8i8i32_aarch64_dotprod()
//     CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  WITH-SVE-DAG:   %[[C8:.+]] = arith.constant 8 : index
//     CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//     CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//     CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//     CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//     CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[$MAP1]]()[%[[K]]]
//         CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//    CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x4xi8>>{%[[TILED_M]], %[[TILED_K]]}
//        NO-SVE:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
///
/// Compute "tiled" N for fixed-width and scalable tiling, see @matmul_lowering_f32f32f32_aarch64.
///
//      WITH-SVE:   %[[VSCALE:.+]] = vector.vscale
//      WITH-SVE:   %[[C8_VSCALE:.+]] = arith.muli %[[VSCALE]], %[[C8]]
//      WITH-SVE:   %[[TILED_N:.+]] = affine.apply #[[$MAP2]]()[%[[N]], %[[C8_VSCALE]]]
//         CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//   NO-SVE-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x4xi8>>{%[[TILED_N]], %[[TILED_K]]}
// WITH-SVE-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?x4xi8>>{%[[TILED_N]], %[[TILED_K]], %[[C8_VSCALE]]}
//         CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//   NO-SVE-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x8x8xi32>>{%[[TILED_M]], %[[TILED_N]]}
// WITH-SVE-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x8x?xi32>>{%[[TILED_M]], %[[TILED_N]], %[[C8_VSCALE]]}
//         CHECK:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]
//    CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 8, 4], strides = [1, 1, 1, 1]
//         CHECK:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]
//   NO-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 8, 4], strides = [1, 1, 1, 1]
// WITH-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], %[[C8_VSCALE]], 4], strides = [1, 1, 1, 1]
//         CHECK:   %[[OUTS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUTS_BINDING]]
//   NO-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
// WITH-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, %[[C8_VSCALE]]], strides = [1, 1, 1, 1]
//         CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//    CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//    CHECK-SAME:       outs(%[[OUTS]] :
//         CHECK:   iree_tensor_ext.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//   NO-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
// WITH-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, %[[C8_VSCALE]]], strides = [1, 1, 1, 1]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_i8i8i32_aarch64_i8mm() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+dotprod,+i8mm,+sve", ukernels = "all", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xi8, #encoding_lhs>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xi8, #encoding_rhs>
  %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
      -> tensor<?x?xi32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #encoding_lhs>,
                   tensor<?x?xi8, #encoding_rhs>)
      outs(%5 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #encoding_result>
      -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  return
}
//     CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//  WITH-SVE-DAG: #[[$MAP1:.+]] = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
//   CHECK-LABEL: func @matmul_lowering_i8i8i32_aarch64_i8mm()
//     CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  WITH-SVE-DAG:   %[[C8:.+]] = arith.constant 8 : index
//     CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//     CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//     CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//     CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//     CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[$MAP0]]()[%[[K]]]
//         CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//    CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x8xi8>>{%[[TILED_M]], %[[TILED_K]]}
//        NO-SVE:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
///
/// Compute "tiled" N for fixed-width and scalable tiling, see @matmul_lowering_f32f32f32_aarch64.
///
//      WITH-SVE:   %[[VSCALE:.+]] = vector.vscale
//      WITH-SVE:   %[[C8_VSCALE:.+]] = arith.muli %[[VSCALE]], %[[C8]]
//      WITH-SVE:   %[[TILED_N:.+]] = affine.apply #[[$MAP1]]()[%[[N]], %[[C8_VSCALE]]]
//         CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//   NO-SVE-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x8xi8>>{%[[TILED_N]], %[[TILED_K]]}
// WITH-SVE-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?x8xi8>>{%[[TILED_N]], %[[TILED_K]], %[[C8_VSCALE]]}
//         CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//   NO-SVE-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x8x8xi32>>{%[[TILED_M]], %[[TILED_N]]}
// WITH-SVE-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x8x?xi32>>{%[[TILED_M]], %[[TILED_N]], %[[C8_VSCALE]]}
//         CHECK:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]
//    CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 8, 8], strides = [1, 1, 1, 1]
//         CHECK:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]
//   NO-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 8, 8], strides = [1, 1, 1, 1]
// WITH-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], %[[C8_VSCALE]], 8], strides = [1, 1, 1, 1]
//         CHECK:   %[[OUTS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUTS_BINDING]]
//   NO-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
// WITH-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, %[[C8_VSCALE]]], strides = [1, 1, 1, 1]
//         CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//    CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//    CHECK-SAME:       outs(%[[OUTS]] :
//         CHECK:   iree_tensor_ext.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//   NO-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
// WITH-SVE-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, %[[C8_VSCALE]]], strides = [1, 1, 1, 1]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_i8i4i32_aarch64() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi4, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xi8, #encoding_lhs>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi4, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xi4, #encoding_rhs>
  %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
      -> tensor<?x?xi32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #encoding_lhs>,
                   tensor<?x?xi4, #encoding_rhs>)
      outs(%5 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #encoding_result>
      -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  return
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
//   CHECK-DAG: #[[$MAP2:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
// CHECK-LABEL: func @matmul_lowering_i8i4i32_aarch64()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//   CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//       CHECK:   %[[TILED_K:.+]] = affine.apply #[[$MAP1]]()[%[[K]]]
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x4x2xi8>>{%[[TILED_M]], %[[TILED_K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP2]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x16x2xi4>>{%[[TILED_N]], %[[TILED_K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x4x16xi32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 4, 2], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 16, 2], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 4, 16], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 4, 16], strides = [1, 1, 1, 1]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_i8i4i32_aarch64_dotprod() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+dotprod", ukernels = "all", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi4, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xi8, #encoding_lhs>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi4, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xi4, #encoding_rhs>
  %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
      -> tensor<?x?xi32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #encoding_lhs>,
                   tensor<?x?xi4, #encoding_rhs>)
      outs(%5 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #encoding_result>
      -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  return
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
// CHECK-LABEL: func @matmul_lowering_i8i4i32_aarch64_dotprod()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//   CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//   CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[$MAP0]]()[%[[K]]]
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x8xi8>>{%[[TILED_M]], %[[TILED_K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP0]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x8xi4>>{%[[TILED_N]], %[[TILED_K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x8x8xi32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 8, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 8, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i4, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_i8i4i32_aarch64_i8mm() attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+dotprod,+i8mm", ukernels = "all", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi4, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xi8, #encoding_lhs>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi4, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xi4, #encoding_rhs>
  %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
      -> tensor<?x?xi32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #encoding_lhs>,
                   tensor<?x?xi4, #encoding_rhs>)
      outs(%5 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #encoding_result>
      -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  return
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//   CHECK-DAG: #[[$MAP2:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
// CHECK-LABEL: func @matmul_lowering_i8i4i32_aarch64_i8mm()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//   CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[$MAP0]]()[%[[M]]]
//   CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[$MAP1]]()[%[[K]]]
//       CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x4x16xi8>>{%[[TILED_M]], %[[TILED_K]]}
//       CHECK:   %[[TILED_N:.+]] = affine.apply #[[$MAP2]]()[%[[N]]]
//       CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x8x16xi4>>{%[[TILED_N]], %[[TILED_K]]}
//       CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x4x8xi32>>{%[[TILED_M]], %[[TILED_N]]}
//       CHECK:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 4, 16], strides = [1, 1, 1, 1]
//       CHECK:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 8, 16], strides = [1, 1, 1, 1]
//       CHECK:   %[[OUTS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 4, 8], strides = [1, 1, 1, 1]
//       CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUTS]] :
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 4, 8], strides = [1, 1, 1, 1]
