// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' --verify-diagnostics --split-input-file %s

#config = #iree_cpu.lowering_config<distribution = [64, 64], vector_common_parallel = [8, 32, 16], vector_reduction = [0, 0, 16], vector_inner_parallel = [0, 0, 0]>
#translation = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
func.func @illegal_parallel_tile_sizes_config(%0: memref<4x8xf32>, %1: memref<8x16xf32>, %2: memref<4x16xf32>) attributes {
  translation_info = #translation
} {
  // expected-error @+1 {{expected only parallel dims to be set in the vector_common_parallel tiling level, but tile size at index (2) was also set}}
  linalg.matmul {lowering_config = #config} ins(%0, %1 : memref<4x8xf32>, memref<8x16xf32>) outs(%2 : memref<4x16xf32>)
  return
}

// -----

#config = #iree_cpu.lowering_config<distribution = [64, 64], vector_common_parallel = [8, 0, 0], vector_reduction = [0, 16, 16], vector_inner_parallel = [0, 0, 0]>
#translation = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
func.func @illegal_reduction_tile_sizes_config(%0: memref<4x8xf32>, %1: memref<8x16xf32>, %2: memref<4x16xf32>) attributes {
  translation_info = #translation
} {
  // expected-error @+1 {{only reduction dims to be set in the vector_reduction tiling level, but tile size at index (1) was also set}}
  linalg.matmul {lowering_config = #config} ins(%0, %1 : memref<4x8xf32>, memref<8x16xf32>) outs(%2 : memref<4x16xf32>)
  return
}

// -----

#config = #iree_cpu.lowering_config<distribution = {sizes = [4, 8], interchange = [1]}, vector_common_parallel = [8, 8, 0], vector_reduction = [0, 0, 8], vector_inner_parallel = [0, 0, 0]>
#translation = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
func.func @illegal_interchange(%0: memref<4x8xf32>, %1: memref<8x16xf32>, %2: memref<4x16xf32>) attributes {
  translation_info = #translation
} {
  // expected-error @+1 {{expected [0, 2) to be set exactly once in interchange for distribution tiling level}}
  linalg.matmul {lowering_config = #config} ins(%0, %1 : memref<4x8xf32>, memref<8x16xf32>) outs(%2 : memref<4x16xf32>)
  return
}

// -----

#config = #iree_cpu.lowering_config<distribution = [0, 7, 7, 64, 0, 0, 0], vector_common_parallel = [6, 1, 7, 32, 0, 0, 0], vector_reduction = [0, 0, 0, 0, 3, 3, 4]>
#translation = #iree_codegen.translation_info<pipeline = CPUConvTileAndDecomposeExpert>
func.func @illegal_conv_config(%0: memref<36x9x9x512xf32>, %1: memref<3x3x512x512xf32>, %2: memref<36x7x7x512xf32>) attributes {
  translation_info = #translation
} {
  // expected-error @+1 {{can't decompose the conv op}}
  linalg.conv_2d_nhwc_hwcf {lowering_config = #config} ins(%0, %1 : memref<36x9x9x512xf32>, memref<3x3x512x512xf32>) outs(%2 : memref<36x7x7x512xf32>)
  return
}

// -----

#config = #iree_cpu.lowering_config<distribution = [0, 7, 7, 64, 0, 0, 0], vector_common_parallel = [6, 1, 7, 32, 0, 0, 0], vector_reduction = [0, 0, 0, 0, [1], 3, 4]>
#translation = #iree_codegen.translation_info<pipeline = CPUConvTileAndDecomposeExpert>
func.func @illegal_conv_config_scalable_unit_dim(%0: memref<36x9x9x512xf32>, %1: memref<3x3x512x512xf32>, %2: memref<36x7x7x512xf32>) attributes {
  translation_info = #translation
} {
  // expected-error @+1 {{can't decompose the conv op}}
  linalg.conv_2d_nhwc_hwcf {lowering_config = #config} ins(%0, %1 : memref<36x9x9x512xf32>, memref<3x3x512x512xf32>) outs(%2 : memref<36x7x7x512xf32>)
  return
}

// -----

#config = #iree_cpu.lowering_config<distribution = [0, 1, 7, 64, 0, 0], vector_common_parallel = [1, 1, 7, 8, 0, 0], vector_reduction = [0, 0, 0, 0, 5, 5]>
#translation = #iree_codegen.translation_info<pipeline = CPUConvTileAndDecomposeExpert>
func.func @illegal_conv_config(%0: memref<1x11x11x576xf32>, %1: memref<5x5x576xf32>, %2: memref<1x7x7x576xf32>) attributes {
  translation_info = #translation
} {
  // expected-error @+1 {{can't decompose the conv op}}
  linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, lowering_config = #config, strides = dense<1> : tensor<2xi64>} ins(%0, %1 : memref<1x11x11x576xf32>, memref<5x5x576xf32>) outs(%2 : memref<1x7x7x576xf32>)
  return
}
