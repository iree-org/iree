// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx950 \
// RUN:   --iree-codegen-llvmgpu-configuration-pipeline \
// RUN:   --iree-codegen-llvmgpu-rocdl-lowering-pipeline %s \
// RUN:   | FileCheck %s

// Verify that a scaled FP4 matmul (mxfp4) lowers all the way to ROCDL
// intrinsics on gfx950 (CDNA4). This catches regressions that partial
// pipeline tests (which stop at mid-level IR) would miss.
//
// TODO: Merge into rocdl_pipeline_test.mlir once that file drops
// --iree-codegen-llvmgpu-use-tile-and-fuse-matmul=false (this test
// requires TileAndFuse for the scaled MFMA matmul path).

#lhs_map = affine_map<(M, N, Ko, Kb) -> (M, Ko, Kb)>
#rhs_map = affine_map<(M, N, Ko, Kb) -> (N, Ko, Kb)>
#scale_m = affine_map<(M, N, Ko, Kb) -> (M, Ko)>
#scale_n = affine_map<(M, N, Ko, Kb) -> (N, Ko)>
#out_map = affine_map<(M, N, Ko, Kb) -> (M, N)>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

func.func @scaled_mxfp4_matmul() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %A_bind = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x4x32xf4E2M1FN>>
  %B_bind = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x4x32xf4E2M1FN>>
  %A_scales_bind = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x4xf8E8M0FNU>>
  %B_scales_bind = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x4xf8E8M0FNU>>
  %C_bind = hal.interface.binding.subspan layout(#pipeline_layout) binding(4) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>

  %A = iree_tensor_ext.dispatch.tensor.load %A_bind, offsets = [0, 0, 0], sizes = [256, 4, 32], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x4x32xf4E2M1FN>> -> tensor<256x4x32xf4E2M1FN>
  %B = iree_tensor_ext.dispatch.tensor.load %B_bind, offsets = [0, 0, 0], sizes = [256, 4, 32], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x4x32xf4E2M1FN>> -> tensor<256x4x32xf4E2M1FN>
  %A_scales = iree_tensor_ext.dispatch.tensor.load %A_scales_bind, offsets = [0, 0], sizes = [256, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x4xf8E8M0FNU>> -> tensor<256x4xf8E8M0FNU>
  %B_scales = iree_tensor_ext.dispatch.tensor.load %B_scales_bind, offsets = [0, 0], sizes = [256, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x4xf8E8M0FNU>> -> tensor<256x4xf8E8M0FNU>

  %empty = tensor.empty() : tensor<256x256xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<256x256xf32>) -> tensor<256x256xf32>

  %result = linalg.generic {
    indexing_maps = [#lhs_map, #rhs_map, #scale_m, #scale_n, #out_map],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]
  } ins(%A, %B, %A_scales, %B_scales : tensor<256x4x32xf4E2M1FN>, tensor<256x4x32xf4E2M1FN>, tensor<256x4xf8E8M0FNU>, tensor<256x4xf8E8M0FNU>)
    outs(%fill : tensor<256x256xf32>) {
  ^bb0(%a: f4E2M1FN, %b: f4E2M1FN, %a_scale: f8E8M0FNU, %b_scale: f8E8M0FNU, %out: f32):
    %0 = arith.scaling_extf %a, %a_scale : f4E2M1FN, f8E8M0FNU to f32
    %1 = arith.scaling_extf %b, %b_scale : f4E2M1FN, f8E8M0FNU to f32
    %2 = arith.mulf %0, %1 : f32
    %3 = arith.addf %out, %2 : f32
    linalg.yield %3 : f32
  } -> tensor<256x256xf32>

  iree_tensor_ext.dispatch.tensor.store %result, %C_bind, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
  return
}

// CHECK-LABEL: llvm.func @scaled_mxfp4_matmul
// CHECK-NOT:     amdgpu.scaled_mfma
// CHECK-COUNT-32: rocdl.mfma.scale.f32.16x16x128.f8f6f4
// CHECK-NOT:     rocdl.mfma.scale.f32.16x16x128.f8f6f4
// CHECK:         llvm.return
