// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-codegen-configuration-preprocessing-pipeline, builtin.module(iree-codegen-spirv-configuration-pipeline, iree-codegen-linalg-to-spirv-pipeline), iree-codegen-translation-postprocessing-pipeline)))" %s \
// RUN: | FileCheck %s

// Smoketest: the full SPIR-V codegen pipeline — variant-scope pre-processing
// (specialize-exports, create-dispatch-config), module-scope configuration and
// linalg-to-SPIR-V lowering, and variant-scope post-processing
// (propagate-dispatch-config) — all drive a simple elementwise add to the
// spirv dialect.

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|fp16|int32, storage = b32|b16, subgroup = shuffle|arithmetic,
    subgroup_size_choices = [32], max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>

#pipeline_layout_static = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

// Static add over 4096 elements.
hal.executable private @add_static {
  hal.executable.variant public @vulkan_spirv_fb target(#executable_target_vulkan_spirv_fb) {
    hal.executable.export public @add_static ordinal(0) layout(#pipeline_layout_static)
        count(%device: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @add_static() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan layout(#pipeline_layout_static) binding(0)
            alignment(64) offset(%c0) flags(ReadOnly)
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf32>>
        %rhs = hal.interface.binding.subspan layout(#pipeline_layout_static) binding(1)
            alignment(64) offset(%c0) flags(ReadOnly)
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf32>>
        %out = hal.interface.binding.subspan layout(#pipeline_layout_static) binding(2)
            alignment(64) offset(%c0)
            : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096xf32>>
        %lhs_t = iree_tensor_ext.dispatch.tensor.load %lhs, offsets = [0], sizes = [4096], strides = [1]
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf32>> -> tensor<4096xf32>
        %rhs_t = iree_tensor_ext.dispatch.tensor.load %rhs, offsets = [0], sizes = [4096], strides = [1]
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf32>> -> tensor<4096xf32>
        %empty = tensor.empty() : tensor<4096xf32>
        %sum = linalg.generic {
            indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
            iterator_types = ["parallel"]}
          ins(%lhs_t, %rhs_t : tensor<4096xf32>, tensor<4096xf32>) outs(%empty : tensor<4096xf32>) {
        ^bb0(%a: f32, %b: f32, %o: f32):
          %s = arith.addf %a, %b : f32
          linalg.yield %s : f32
        } -> tensor<4096xf32>
        iree_tensor_ext.dispatch.tensor.store %sum, %out, offsets = [0], sizes = [4096], strides = [1]
            : tensor<4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096xf32>>
        return
      }
    }
  }
}
// CHECK-LABEL: hal.executable private @add_static
//       CHECK:   hal.executable.export public @add_static
//       CHECK:     hal.return
//       CHECK:   builtin.module
//       CHECK:     spirv.module
//       CHECK:       spirv.func @add_static
//       CHECK:         spirv.FAdd
//   CHECK-NOT:   iree_codegen.dispatch_config

// -----

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|fp16|int32, storage = b32|b16, subgroup = shuffle|arithmetic,
    subgroup_size_choices = [32], max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>

#pipeline_layout_dynamic = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

// Dynamic add: workload dim threaded through a workload.ordinal, with a matching
// count region on the export.
hal.executable private @add_dynamic {
  hal.executable.variant public @vulkan_spirv_fb target(#executable_target_vulkan_spirv_fb) {
    hal.executable.export public @add_dynamic ordinal(0) layout(#pipeline_layout_dynamic)
        count(%device: !hal.device, %w0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%w0)
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @add_dynamic() {
        %c0 = arith.constant 0 : index
        %dim_i32 = hal.interface.constant.load layout(#pipeline_layout_dynamic) ordinal(0) : i32
        %dim = arith.index_castui %dim_i32 : i32 to index
        %n = iree_tensor_ext.dispatch.workload.ordinal %dim, 0 : index
        %lhs = hal.interface.binding.subspan layout(#pipeline_layout_dynamic) binding(0)
            alignment(64) offset(%c0) flags(ReadOnly)
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf32>>{%n}
        %rhs = hal.interface.binding.subspan layout(#pipeline_layout_dynamic) binding(1)
            alignment(64) offset(%c0) flags(ReadOnly)
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf32>>{%n}
        %out = hal.interface.binding.subspan layout(#pipeline_layout_dynamic) binding(2)
            alignment(64) offset(%c0)
            : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%n}
        %lhs_t = iree_tensor_ext.dispatch.tensor.load %lhs, offsets = [0], sizes = [%n], strides = [1]
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf32>>{%n} -> tensor<?xf32>
        %rhs_t = iree_tensor_ext.dispatch.tensor.load %rhs, offsets = [0], sizes = [%n], strides = [1]
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf32>>{%n} -> tensor<?xf32>
        %empty = tensor.empty(%n) : tensor<?xf32>
        %sum = linalg.generic {
            indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
            iterator_types = ["parallel"]}
          ins(%lhs_t, %rhs_t : tensor<?xf32>, tensor<?xf32>) outs(%empty : tensor<?xf32>) {
        ^bb0(%a: f32, %b: f32, %o: f32):
          %s = arith.addf %a, %b : f32
          linalg.yield %s : f32
        } -> tensor<?xf32>
        iree_tensor_ext.dispatch.tensor.store %sum, %out, offsets = [0], sizes = [%n], strides = [1]
            : tensor<?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%n}
        return
      }
    }
  }
}
// CHECK-LABEL: hal.executable private @add_dynamic
//       CHECK:   hal.executable.export public @add_dynamic
//       CHECK:     hal.return
//       CHECK:   builtin.module
//       CHECK:     spirv.module
//       CHECK:       spirv.func @add_dynamic
//       CHECK:         spirv.FAdd
//   CHECK-NOT:   iree_codegen.dispatch_config
