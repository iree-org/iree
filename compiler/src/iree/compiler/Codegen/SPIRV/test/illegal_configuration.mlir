// RUN: iree-opt \
// RUN:   --pass-pipeline='builtin.module(iree-codegen-materialize-user-configs, iree-spirv-select-lowering-strategy-pass)' \
// RUN:   --verify-diagnostics --split-input-file %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_codegen.lowering_config<tile_sizes = []>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#translation = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize workgroup_size = [16, 8, 1], {pipeline_depth = 0 : i64, store_stage = 1 : i64}>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
func.func @illegal() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<4x8xf32>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<8x16xf32>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : memref<4x16xf32>
  // expected-error @+1 {{expected 1 levels of tiling sizes, got 0}}
  linalg.matmul {compilation_info = #compilation} ins(%0, %1 : memref<4x8xf32>, memref<8x16xf32>) outs(%2 : memref<4x16xf32>)
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[32, 64], [4, 4], [0, 0, 4]]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#translation = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize, {pipeline_depth = 0 : i64, store_stage = 1 : i64}>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
// expected-error @+1 {{expected workgroup size to have three dimensions for SPIR-V pipelines}}
func.func @illegal() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<64x16xf32>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<16x128xf32>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : memref<64x128xf32>
  linalg.matmul {compilation_info = #compilation} ins(%0, %1 : memref<64x16xf32>, memref<16x128xf32>) outs(%2 : memref<64x128xf32>)
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[32, 64], [4, 4], [0, 0, 4]]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#translation = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize workgroup_size = [16, 8, 128], {pipeline_depth = 0 : i64, store_stage = 1 : i64}>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
func.func @illegal() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<64x16xf32>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<16x128xf32>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : memref<64x128xf32>
  // expected-error @+1 {{expected workgroup size dimensions not exceeding [128, 128, 64]}}
  linalg.matmul {compilation_info = #compilation} ins(%0, %1 : memref<64x16xf32>, memref<16x128xf32>) outs(%2 : memref<64x128xf32>)
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[32, 64], [4, 2], [0, 0, 4]]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#translation = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize workgroup_size = [32, 8, 1], {pipeline_depth = 0 : i64, store_stage = 1 : i64}>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
func.func @illegal() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<64x16xf32>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<16x128xf32>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : memref<64x128xf32>
  // expected-error @+1 {{expected total invocation count in workgroup to be <= 128}}
  linalg.matmul {compilation_info = #compilation} ins(%0, %1 : memref<64x16xf32>, memref<16x128xf32>) outs(%2 : memref<64x128xf32>)
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[32, 64], [16, 8], [0, 0, 4]]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#translation = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize workgroup_size = [8, 2, 1], {pipeline_depth = 0 : i64, store_stage = 1 : i64}>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
func.func @illegal() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<64x16xf32>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<16x128xf32>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : memref<64x128xf32>
  // expected-error @+1 {{expected total workgroup size to be multiple of 32}}
  linalg.matmul {compilation_info = #compilation} ins(%0, %1 : memref<64x16xf32>, memref<16x128xf32>) outs(%2 : memref<64x128xf32>)
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[32, 60], [4, 4], [0, 0, 4]]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#translation = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize workgroup_size = [15, 8, 1], {pipeline_depth = 0 : i64, store_stage = 1 : i64}>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
func.func @illegal() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<64x16xf32>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<16x128xf32>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : memref<64x128xf32>
  // expected-error @+1 {{expected each workgroup size dimension to be power of two}}
  linalg.matmul {compilation_info = #compilation} ins(%0, %1 : memref<64x16xf32>, memref<16x128xf32>) outs(%2 : memref<64x128xf32>)
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[32, 64, 4]]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#translation = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize workgroup_size = [16, 8, 1], {pipeline_depth = 0 : i64, store_stage = 1 : i64}>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
func.func @illegal() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<48x16xf32>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<16x128xf32>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : memref<48x128xf32>
  // expected-error @+1 {{LHS shape is indivisible by first level tile size}}
  linalg.matmul {compilation_info = #compilation} ins(%0, %1 : memref<48x16xf32>, memref<16x128xf32>) outs(%2 : memref<48x128xf32>)
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[32, 64, 4]]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#translation = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize workgroup_size = [16, 8, 1], {pipeline_depth = 0 : i64, store_stage = 1 : i64}>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
func.func @illegal() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<64x16xf32>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<16x80xf32>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : memref<64x80xf32>
  // expected-error @+1 {{RHS shape is indivisible by first level tile size}}
  linalg.matmul {compilation_info = #compilation} ins(%0, %1 : memref<64x16xf32>, memref<16x80xf32>) outs(%2 : memref<64x80xf32>)
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [32, 32], [0, 0, 16]]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|fp16|int32, storage = b32|b16, subgroup = none, dot = none,
    mma = [<WMMA_F32_16x16x16_F16>, <WMMA_F16_16x16x16_F16>],
    subgroup_size_choices = [32, 64], max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#translation = #iree_codegen.translation_info<SPIRVCooperativeMatrixVectorize workgroup_size = [128, 2, 1] subgroup_size = 64, {pipeline_depth = 0 : i64, store_stage = 1 : i64}>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
func.func @matmul_tensor() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<64x32xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<32x128xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<64x128xf16>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [64, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x32xf16>> -> tensor<64x32xf16>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x128xf16>> -> tensor<32x128xf16>
  %5 = tensor.empty() : tensor<64x128xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<64x128xf16>) -> tensor<64x128xf16>
  // expected-error @+1 {{expected 4 levels of tiling sizes, got 3}}
  %7 = linalg.matmul {compilation_info = #compilation} ins(%3, %4 : tensor<64x32xf16>, tensor<32x128xf16>) outs(%6 : tensor<64x128xf16>) -> tensor<64x128xf16>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [64, 128], strides = [1, 1] : tensor<64x128xf16> -> !flow.dispatch.tensor<writeonly:tensor<64x128xf16>>
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [32, 32], [0, 0, 16], [8, 8, 8]]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|fp16|int32, storage = b32|b16, subgroup = none, dot = none,
    mma = [<WMMA_F32_16x16x16_F16>, <WMMA_F16_16x16x16_F16>],
    subgroup_size_choices = [32, 64], max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#translation = #iree_codegen.translation_info<SPIRVCooperativeMatrixVectorize workgroup_size = [128, 2, 1] subgroup_size = 64, {pipeline_depth = 0 : i64, store_stage = 1 : i64}>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
func.func @matmul_tensor() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<64x32xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<32x128xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<64x128xf16>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [64, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x32xf16>> -> tensor<64x32xf16>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x128xf16>> -> tensor<32x128xf16>
  %5 = tensor.empty() : tensor<64x128xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<64x128xf16>) -> tensor<64x128xf16>
  // expected-error @+1 {{expected the fourth level tile sizes to match cooperative matrix sizes}}
  %7 = linalg.matmul {compilation_info = #compilation} ins(%3, %4 : tensor<64x32xf16>, tensor<32x128xf16>) outs(%6 : tensor<64x128xf16>) -> tensor<64x128xf16>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [64, 128], strides = [1, 1] : tensor<64x128xf16> -> !flow.dispatch.tensor<writeonly:tensor<64x128xf16>>
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[32, 32], [8, 8], [0, 0, 4], [16, 16, 16]]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|fp16|int32, storage = b32|b16, subgroup = none, dot = none,
    mma = [<WMMA_F32_16x16x16_F16>, <WMMA_F16_16x16x16_F16>],
    subgroup_size_choices = [32, 64], max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#translation = #iree_codegen.translation_info<SPIRVCooperativeMatrixVectorize workgroup_size = [256, 4, 1] subgroup_size = 64, {pipeline_depth = 0 : i64, store_stage = 1 : i64}>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
func.func @matmul_tensor() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<64x32xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<32x128xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<64x128xf16>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [64, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x32xf16>> -> tensor<64x32xf16>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x128xf16>> -> tensor<32x128xf16>
  %5 = tensor.empty() : tensor<64x128xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<64x128xf16>) -> tensor<64x128xf16>
  // expected-error @+1 {{expected subgroup tile sizes to be multiple of [16, 16, 16]}}
  %7 = linalg.matmul {compilation_info = #compilation} ins(%3, %4 : tensor<64x32xf16>, tensor<32x128xf16>) outs(%6 : tensor<64x128xf16>) -> tensor<64x128xf16>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [64, 128], strides = [1, 1] : tensor<64x128xf16> -> !flow.dispatch.tensor<writeonly:tensor<64x128xf16>>
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [32, 32], [0, 0, 16], [16, 16, 16]]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|fp16|int32, storage = b32|b16, subgroup = none, dot = none,
    mma = [<WMMA_F32_16x16x16_F16>, <WMMA_F16_16x16x16_F16>],
    subgroup_size_choices = [32, 64], max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#translation = #iree_codegen.translation_info<SPIRVCooperativeMatrixVectorize workgroup_size = [64, 2, 1] subgroup_size = 64, {pipeline_depth = 0 : i64, store_stage = 1 : i64}>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
func.func @matmul_tensor() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<64x32xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<32x128xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<64x128xf16>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [64, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x32xf16>> -> tensor<64x32xf16>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x128xf16>> -> tensor<32x128xf16>
  %5 = tensor.empty() : tensor<64x128xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<64x128xf16>) -> tensor<64x128xf16>
  // expected-error @+1 {{expected workgroup x component equals to (warp_size * wg_tile_n / subgroup_tile_n)}}
  %7 = linalg.matmul {compilation_info = #compilation} ins(%3, %4 : tensor<64x32xf16>, tensor<32x128xf16>) outs(%6 : tensor<64x128xf16>) -> tensor<64x128xf16>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [64, 128], strides = [1, 1] : tensor<64x128xf16> -> !flow.dispatch.tensor<writeonly:tensor<64x128xf16>>
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [32, 32], [0, 0, 16], [16, 16, 16]]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|fp16|int32, storage = b32|b16, subgroup = none, dot = none,
    mma = [<WMMA_F32_16x16x16_F16>, <WMMA_F16_16x16x16_F16>],
    subgroup_size_choices = [32, 64], max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#translation = #iree_codegen.translation_info<SPIRVCooperativeMatrixVectorize workgroup_size = [128, 4, 1] subgroup_size = 64, {pipeline_depth = 0 : i64, store_stage = 1 : i64}>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
func.func @matmul_tensor() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<64x32xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<32x128xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<64x128xf16>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [64, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x32xf16>> -> tensor<64x32xf16>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x128xf16>> -> tensor<32x128xf16>
  %5 = tensor.empty() : tensor<64x128xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<64x128xf16>) -> tensor<64x128xf16>
  // expected-error @+1 {{expected workgroup y component equals to (wg_tile_m / subgroup_tile_m)}}
  %7 = linalg.matmul {compilation_info = #compilation} ins(%3, %4 : tensor<64x32xf16>, tensor<32x128xf16>) outs(%6 : tensor<64x128xf16>) -> tensor<64x128xf16>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [64, 128], strides = [1, 1] : tensor<64x128xf16> -> !flow.dispatch.tensor<writeonly:tensor<64x128xf16>>
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[0, 4, 4, 16], [0, 2, 2, 2], [0, 0, 0, 0, 1, 1, 4]]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<()[s0] -> (s0 * 4)>
#map1 = affine_map<()[s0] -> (s0 * 16)>
#map2 = affine_map<(d0) -> (d0 * 2)>
#translation = #iree_codegen.translation_info<SPIRVBaseVectorize workgroup_size = [8, 2, 2]>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
func.func @illegal() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c112 = arith.constant 112 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x225x225x8xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<3x3x8x16xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %workgroup_id_z = hal.interface.workgroup.id[2] : index
  %workgroup_count_z = hal.interface.workgroup.count[2] : index
  %3 = affine.apply #map()[%workgroup_id_z]
  %4 = affine.apply #map()[%workgroup_count_z]
  scf.for %arg0 = %3 to %c112 step %4 {
    %5 = affine.apply #map()[%workgroup_id_y]
    %6 = affine.apply #map()[%workgroup_count_y]
    scf.for %arg1 = %5 to %c112 step %6 {
      %7 = affine.apply #map1()[%workgroup_id_x]
      %8 = affine.apply #map1()[%workgroup_count_x]
      scf.for %arg2 = %7 to %c16 step %8 {
        %9 = flow.dispatch.tensor.load %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, 4, 4, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>> -> tensor<1x4x4x16xf32>
        %10 = affine.apply #map2(%arg0)
        %11 = affine.apply #map2(%arg1)
        %12 = flow.dispatch.tensor.load %0, offsets = [0, %10, %11, 0], sizes = [1, 9, 9, 8], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x225x225x8xf32>> -> tensor<1x9x9x8xf32>
        %13 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, %arg2], sizes = [3, 3, 8, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x8x16xf32>> -> tensor<3x3x8x16xf32>
        %14 = linalg.fill ins(%cst : f32) outs(%9 : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
        // expected-error @+1 {{expected 4 levels of tiling sizes, got 3}}
        %15 = linalg.conv_2d_nhwc_hwcf {compilation_info = #compilation, dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%12, %13 : tensor<1x9x9x8xf32>, tensor<3x3x8x16xf32>) outs(%14 : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
        flow.dispatch.tensor.store %15, %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, 4, 4, 16], strides = [1, 1, 1, 1] : tensor<1x4x4x16xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>>
      }
    }
  }
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[0, 6, 6, 16], [0, 3, 3, 2], [0, 0, 0, 0, 1, 1, 4], [0, 1, 0, 0]]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<()[s0] -> (s0 * 4)>
#map1 = affine_map<()[s0] -> (s0 * 16)>
#map2 = affine_map<(d0) -> (d0 * 2)>
#translation = #iree_codegen.translation_info<SPIRVBaseVectorize workgroup_size = [8, 2, 2]>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
func.func @illegal() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c112 = arith.constant 112 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x225x225x8xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<3x3x8x16xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %workgroup_id_z = hal.interface.workgroup.id[2] : index
  %workgroup_count_z = hal.interface.workgroup.count[2] : index
  %3 = affine.apply #map()[%workgroup_id_z]
  %4 = affine.apply #map()[%workgroup_count_z]
  scf.for %arg0 = %3 to %c112 step %4 {
    %5 = affine.apply #map()[%workgroup_id_y]
    %6 = affine.apply #map()[%workgroup_count_y]
    scf.for %arg1 = %5 to %c112 step %6 {
      %7 = affine.apply #map1()[%workgroup_id_x]
      %8 = affine.apply #map1()[%workgroup_count_x]
      scf.for %arg2 = %7 to %c16 step %8 {
        %9 = flow.dispatch.tensor.load %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, 4, 4, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>> -> tensor<1x4x4x16xf32>
        %10 = affine.apply #map2(%arg0)
        %11 = affine.apply #map2(%arg1)
        %12 = flow.dispatch.tensor.load %0, offsets = [0, %10, %11, 0], sizes = [1, 9, 9, 8], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x225x225x8xf32>> -> tensor<1x9x9x8xf32>
        %13 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, %arg2], sizes = [3, 3, 8, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x8x16xf32>> -> tensor<3x3x8x16xf32>
        %14 = linalg.fill ins(%cst : f32) outs(%9 : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
        // expected-error @+1 {{expected first level tile size divides the output size [OH, OW, OC]}}
        %15 = linalg.conv_2d_nhwc_hwcf {compilation_info = #compilation, dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%12, %13 : tensor<1x9x9x8xf32>, tensor<3x3x8x16xf32>) outs(%14 : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
        flow.dispatch.tensor.store %15, %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, 4, 4, 16], strides = [1, 1, 1, 1] : tensor<1x4x4x16xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>>
      }
    }
  }
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[0, 4, 4, 16], [0, 2, 2, 4], [0, 0, 0, 0, 1, 1, 4], [0, 1, 0, 0]]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<()[s0] -> (s0 * 4)>
#map1 = affine_map<()[s0] -> (s0 * 16)>
#map2 = affine_map<(d0) -> (d0 * 2)>
#translation = #iree_codegen.translation_info<SPIRVBaseVectorize workgroup_size = [8, 2, 2]>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
func.func @illegal() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c112 = arith.constant 112 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x225x225x8xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<3x3x8x16xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %workgroup_id_z = hal.interface.workgroup.id[2] : index
  %workgroup_count_z = hal.interface.workgroup.count[2] : index
  %3 = affine.apply #map()[%workgroup_id_z]
  %4 = affine.apply #map()[%workgroup_count_z]
  scf.for %arg0 = %3 to %c112 step %4 {
    %5 = affine.apply #map()[%workgroup_id_y]
    %6 = affine.apply #map()[%workgroup_count_y]
    scf.for %arg1 = %5 to %c112 step %6 {
      %7 = affine.apply #map1()[%workgroup_id_x]
      %8 = affine.apply #map1()[%workgroup_count_x]
      scf.for %arg2 = %7 to %c16 step %8 {
        %9 = flow.dispatch.tensor.load %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, 4, 4, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>> -> tensor<1x4x4x16xf32>
        %10 = affine.apply #map2(%arg0)
        %11 = affine.apply #map2(%arg1)
        %12 = flow.dispatch.tensor.load %0, offsets = [0, %10, %11, 0], sizes = [1, 9, 9, 8], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x225x225x8xf32>> -> tensor<1x9x9x8xf32>
        %13 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, %arg2], sizes = [3, 3, 8, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x8x16xf32>> -> tensor<3x3x8x16xf32>
        %14 = linalg.fill ins(%cst : f32) outs(%9 : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
        // expected-error @+1 {{expected workgroup tile sizes to be the product of thread tile size and workgroup size}}
        %15 = linalg.conv_2d_nhwc_hwcf {compilation_info = #compilation, dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%12, %13 : tensor<1x9x9x8xf32>, tensor<3x3x8x16xf32>) outs(%14 : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
        flow.dispatch.tensor.store %15, %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, 4, 4, 16], strides = [1, 1, 1, 1] : tensor<1x4x4x16xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>>
      }
    }
  }
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[0, 1, 7, 64], [0, 1, 7, 2], [0, 0, 0, 0, 5, 5], [0, 1, 0, 0]]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#translation = #iree_codegen.translation_info<SPIRVBaseVectorize workgroup_size = [32, 1, 1]>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
func.func @illegal() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<1x11x11x576xf32>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<5x5x576xf32>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : memref<1x7x7x576xf32>
  // expected-error @+1 {{expected tile sizes for KH and KW to be 1}}
  linalg.depthwise_conv_2d_nhwc_hwc {compilation_info = #compilation} ins(%0, %1 : memref<1x11x11x576xf32>, memref<5x5x576xf32>) outs(%2 : memref<1x7x7x576xf32>)
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[0, 1, 7, 64], [0, 1, 7, 2], [0, 0, 0, 0, 1, 1], [0, 0, 1, 1]]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [],
    subgroup_size_choices = [32], max_workgroup_sizes = [128, 128, 64],
    max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#translation = #iree_codegen.translation_info<SPIRVBaseVectorize workgroup_size = [32, 1, 1]>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
func.func @illegal() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<1x11x11x576xf32>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<5x5x576xf32>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : memref<1x7x7x576xf32>
  // expected-error @+1 {{expected the fourth level of tile size to be [0, 1, 0, 0]}}
  linalg.depthwise_conv_2d_nhwc_hwc {compilation_info = #compilation} ins(%0, %1 : memref<1x11x11x576xf32>, memref<5x5x576xf32>) outs(%2 : memref<1x7x7x576xf32>)
  return
}
