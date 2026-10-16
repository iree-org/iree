// RUN: iree-opt --iree-gpu-test-target=gfx1250 --iree-convert-to-rocdl --verify-diagnostics %s

// gfx1250 only runs wave32, so a wave64 subgroup size cannot be represented in
// the target description and is rejected up front.
#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute>
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

// expected-error @+1 {{target only supports a wavefront size of 32}}
module {
  func.func @wave64_on_wave32_only_target() attributes {translation_info = #translation} {
    return
  }
}
