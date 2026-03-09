// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-codegen-decompose-softmax), iree-spirv-select-lowering-strategy-pass)' %s | \
// RUN:   FileCheck %s

// Verifies that for decomposed softmax (max-reduce, exp-sum-reduce, div), the
// lowering config is placed on the last reduction (exp-sum) rather than the
// first (max).

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = none,
    subgroup_size_choices = [64], max_workgroup_sizes = [512, 512, 512],
    max_thread_count_per_workgroup = 512, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
func.func @softmax(%arg0: tensor<10x256x256xf32>) -> tensor<10x256x256xf32>
    attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %0 = tensor.empty() : tensor<10x256x256xf32>
  %1 = linalg.softmax dimension(2)
      ins(%arg0 : tensor<10x256x256xf32>)
      outs(%0 : tensor<10x256x256xf32>) -> tensor<10x256x256xf32>
  return %1 : tensor<10x256x256xf32>
}

// The lowering_config should be on the exp-sum reduction (the second generic
// with a reduction iterator), not on the max reduction (the first).

// CHECK-LABEL: func.func @softmax
// Max reduction: no lowering_config.
//       CHECK:   linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "reduction"]
//   CHECK-NOT:       lowering_config
// Exp-sum reduction: has lowering_config.
//       CHECK:   linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME:       lowering_config
// Div elementwise: no lowering_config.
//       CHECK:   linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel"]
//   CHECK-NOT:       lowering_config
