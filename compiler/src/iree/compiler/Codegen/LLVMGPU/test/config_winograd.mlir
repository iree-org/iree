// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1100 --pass-pipeline='builtin.module(iree-llvmgpu-select-lowering-strategy)' %s | FileCheck %s

func.func @winograd_filter_transform(%2: tensor<3x3x64x128xf32>) -> tensor<8x8x64x128xf32> {
  %3 = tensor.empty() : tensor<8x8x64x128xf32>
  %4 = iree_linalg_ext.winograd.filter_transform output_tile_size(6) kernel_size(3) kernel_dimensions([0, 1]) ins(%2 : tensor<3x3x64x128xf32>) outs(%3 : tensor<8x8x64x128xf32>) -> tensor<8x8x64x128xf32>
  return %4 : tensor<8x8x64x128xf32>
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32, 16], [1, 1]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUWinogradVectorize workgroup_size = [16, 32, 1]>
//       CHECK: func.func @winograd_filter_transform(
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   iree_linalg_ext.winograd.filter_transform
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

func.func @winograd_input_transform(%2: tensor<2x34x34x128xf16>) -> tensor<8x8x2x6x6x128xf16> {
  %3 = tensor.empty() : tensor<8x8x2x6x6x128xf16>
  %4 = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2]) ins(%2 : tensor<2x34x34x128xf16>) outs(%3 : tensor<8x8x2x6x6x128xf16>) -> tensor<8x8x2x6x6x128xf16>
  return %4 : tensor<8x8x2x6x6x128xf16>
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 4, 4, 32], [1, 1, 1, 1]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUWinogradVectorize workgroup_size = [32, 4, 4]>
//       CHECK: func.func @winograd_input_transform(
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   iree_linalg_ext.winograd.input_transform
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

func.func @winograd_output_transform(%2: tensor<8x8x2x6x6x128xf16>) -> tensor<2x36x36x128xf16> {
  %3 = tensor.empty() : tensor<2x36x36x128xf16>
  %4 = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2]) ins(%2 : tensor<8x8x2x6x6x128xf16>) outs(%3 : tensor<2x36x36x128xf16>) -> tensor<2x36x36x128xf16>
  return %4 : tensor<2x36x36x128xf16>
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 4, 4, 32], [1, 1, 1, 1]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUWinogradVectorize workgroup_size = [32, 4, 4]>
//       CHECK: func.func @winograd_output_transform(
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   iree_linalg_ext.winograd.output_transform
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]
