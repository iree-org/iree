// RUN: iree-opt --pass-pipeline="builtin.module(iree-codegen-llvmgpu-configuration-pipeline)" \
// RUN:    --iree-llvmgpu-test-combine-layout-transformation=true \
// RUN:    --iree-gpu-test-target=gfx942 --split-input-file %s | FileCheck %s

// Verify that relayout ops are propagated to the end of the dispatch, and then
// combined, with the compute op at the beginning of the dispatch. Also verify
// that padding values are written separately from the tensor part of the
// kernel.

#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @relayout_ops_with_compute_between() {
  %cst = arith.constant 0.000000e+00 : f16
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x16x4x8x4x4x16x2xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x63x8x4x4x4x2x4xf16>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0, 0, 0, 0, 0], sizes = [16, 16, 4, 8, 4, 4, 16, 2], strides = [1, 1, 1, 1, 1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x16x4x8x4x4x16x2xf32>> -> tensor<16x16x4x8x4x4x16x2xf32>
  %collapsed = tensor.collapse_shape %2 [[0], [1], [2, 3, 4], [5, 6, 7]]
      : tensor<16x16x4x8x4x4x16x2xf32> into tensor<16x16x128x128xf32>
  %3 = tensor.empty() : tensor<2000x2000xf32>
  %unpack = linalg.unpack %collapsed
      outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [128, 128]
      into %3 : tensor<16x16x128x128xf32> -> tensor<2000x2000xf32>
  %4 = tensor.empty() : tensor<2000x2000xf16>
  %5 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%unpack : tensor<2000x2000xf32>) outs(%4 : tensor<2000x2000xf16>) {
  ^bb0(%in: f32, %out: f16):
    %14 = arith.truncf %in : f32 to f16
    linalg.yield %14 : f16
  } -> tensor<2000x2000xf16>
  %6 = tensor.empty() : tensor<16x63x128x32xf16>
  %pack = linalg.pack %5 padding_value(%cst : f16)
      outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [128, 32]
      into %6 : tensor<2000x2000xf16> -> tensor<16x63x128x32xf16>
  %expanded = tensor.expand_shape %pack [[0], [1], [2, 3, 4], [5, 6, 7]] output_shape [16, 63, 4, 8, 4, 2, 4, 4]
      : tensor<16x63x128x32xf16> into tensor<16x63x4x8x4x2x4x4xf16>
  %7 = tensor.empty() : tensor<16x63x8x4x4x4x2x4xf16>
  %transposed = linalg.transpose ins(%expanded : tensor<16x63x4x8x4x2x4x4xf16>) outs(%7 : tensor<16x63x8x4x4x4x2x4xf16>) permutation = [0, 1, 3, 6, 2, 4, 5, 7]
  iree_tensor_ext.dispatch.tensor.store %transposed, %1, offsets = [0, 0, 0, 0, 0, 0, 0, 0], sizes = [16, 63, 8, 4, 4, 4, 2, 4], strides = [1, 1, 1, 1, 1, 1, 1, 1] : tensor<16x63x8x4x4x4x2x4xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x63x8x4x4x4x2x4xf16>>
  return
}
// CHECK:      @relayout_ops_with_compute_between()
// CHECK:        %[[PAD_VAL:.+]] = arith.constant 0.000000e+00 : f16
// CHECK:        %[[SRC_SUBSPAN:.+]] = hal.interface.binding.subspan{{.*}} binding(0)
// CHECK:        %[[SRC_BUFFER:.+]] = amdgpu.fat_raw_buffer_cast %[[SRC_SUBSPAN]]
// CHECK:        %[[DEST_SUBSPAN:.+]] = hal.interface.binding.subspan{{.*}} binding(1)
// CHECK:        %[[DEST_BUFFER:.+]] = amdgpu.fat_raw_buffer_cast %[[DEST_SUBSPAN]]
// CHECK:        %[[SRC:.+]] = iree_codegen.load_from_buffer %[[SRC_BUFFER]]
// CHECK:        %[[COMPUTE_OP:.+]] = linalg.generic{{.*}} ins(%[[SRC]]
// CHECK-NEXT:   ^bb0
// CHECK-NEXT:     arith.truncf
// CHECK:        %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter %[[COMPUTE_OP]]
// CHECK:        iree_codegen.store_to_buffer %[[MAP_SCATTER]], %[[DEST_BUFFER]]
// CHECK:        scf.forall
// CHECK:          scf.forall
// CHECK:            scf.if
// CHECK-NEXT:         memref.store %[[PAD_VAL]], %[[DEST_BUFFER]]
// CHECK:          } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
// CHECK:        } {mapping = [#iree_codegen.workgroup_mapping<x>, #iree_codegen.workgroup_mapping<y>]}
