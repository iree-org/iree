// RUN: iree-opt --iree-gpu-test-target=gfx942 --pass-pipeline='builtin.module(iree-llvmgpu-select-lowering-strategy)' --verify-diagnostics --split-input-file %s

#config = #iree_gpu.lowering_config<{reduction = [0, 0, 32, 1, 0]}>
#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [64, 1, 1] subgroup_size = 64>
func.func @too_many_reduction_tile_sizes(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>, %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {translation_info = #translation} {
  // expected-error @+1 {{expected no more than 3 tile sizes in the reduction tiling level, but 5 were set}}
  %0 = linalg.matmul {lowering_config = #config}
       ins(%a, %b : tensor<64x64xf32>, tensor<64x64xf32>)
       outs(%c : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// -----

#config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0, 1, 0]}>
#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [64, 1, 1] subgroup_size = 64>
func.func @too_many_workgroup_tile_sizes(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>, %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {translation_info = #translation} {
  // expected-error @+1 {{expected no more than 3 tile sizes in the workgroup tiling level, but 5 were set}}
  %0 = linalg.matmul {lowering_config = #config}
       ins(%a, %b : tensor<64x64xf32>, tensor<64x64xf32>)
       outs(%c : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// -----

#config = #iree_gpu.lowering_config<{reduction = [32, 0, 0]}>
#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [64, 1, 1] subgroup_size = 64>
func.func @reduction_tile_on_parallel_dim(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>, %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {translation_info = #translation} {
  // expected-error @+1 {{expected only reduction dims to be set in the reduction tiling level, but tile size at index (0) was also set}}
  %0 = linalg.matmul {lowering_config = #config}
       ins(%a, %b : tensor<64x64xf32>, tensor<64x64xf32>)
       outs(%c : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// -----

#config = #iree_gpu.lowering_config<{workgroup = [0, 0, 32]}>
#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [64, 1, 1] subgroup_size = 64>
func.func @workgroup_tile_on_reduction_dim(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>, %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {translation_info = #translation} {
  // expected-error @+1 {{expected only parallel dims to be set in the workgroup tiling level, but tile size at index (2) was also set}}
  %0 = linalg.matmul {lowering_config = #config}
       ins(%a, %b : tensor<64x64xf32>, tensor<64x64xf32>)
       outs(%c : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}
