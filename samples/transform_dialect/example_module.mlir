// Source IR for the following. Skips dispatch formation to isolate testing to
// codegen.
//
// !A_size = tensor<16x5xf32>
// !B_size = tensor<5x16xf32>
// !C_size = tensor<16x16xf32>
// !O_size = tensor<16xf32>
//
// module {
//   func.func @example_module(%A : !A_size, %B : !B_size, %C : !C_size) -> !O_size {
//     %0 = linalg.add ins(%A, %A : !A_size, !A_size)
//                     outs(%A : !A_size) -> !A_size
//     %1 = linalg.matmul ins(%0, %B : !A_size, !B_size)
//                        outs(%C : !C_size) -> !C_size
//     %empty = tensor.empty() : !O_size
//     %2 = linalg.reduce
//       ins(%1 : !C_size)
//       outs(%empty : !O_size)
//       dimensions = [1]
//       (%in: f32, %out: f32) {
//         %3 = arith.addf %out, %in: f32
//         linalg.yield %3: f32
//       }
//     return %2 : !O_size
//   }
// }

#target = #iree_gpu.target<arch = "", features = "spirv:v1.3,cap:Shader", wgp = <
  compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [64, 64],
  max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>

#pipeline_layout_0 = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>
#pipeline_layout_1 = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>
#pipeline_layout_2 = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>

module attributes {
  hal.device.targets = [
    #hal.device.target<"vulkan", [
      #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
        iree.gpu.target = #target
      }>
    ]> : !hal.device
  ]
} {
  hal.executable private @example_module_dispatch_0 {
    hal.executable.variant public @vulkan_spirv_fb target(<"vulkan-spirv", "vulkan-spirv-fb", {iree.gpu.target = #target}>) {
      hal.executable.export public @example_module_dispatch_0_generic_80_f32 ordinal(0) layout(#pipeline_layout_0) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @example_module_dispatch_0_generic_80_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan layout(#pipeline_layout_0) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<80xf32>>
          %1 = hal.interface.binding.subspan layout(#pipeline_layout_0) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<80xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [80], strides = [1] : !flow.dispatch.tensor<readonly:tensor<80xf32>> -> tensor<80xf32>
          %3 = tensor.empty() : tensor<80xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2 : tensor<80xf32>) outs(%3 : tensor<80xf32>) {
          ^bb0(%in: f32, %out: f32):
            %5 = arith.addf %in, %in : f32
            linalg.yield %5 : f32
          } -> tensor<80xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0], sizes = [80], strides = [1] : tensor<80xf32> -> !flow.dispatch.tensor<writeonly:tensor<80xf32>>
          return
        }
      }
    }
  }
  hal.executable private @example_module_dispatch_1 {
    hal.executable.variant public @vulkan_spirv_fb target(<"vulkan-spirv", "vulkan-spirv-fb", {iree.gpu.target = #target}>) {
      hal.executable.export public @example_module_dispatch_1_matmul_16x16x5_f32 ordinal(0) layout(#pipeline_layout_1) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @example_module_dispatch_1_matmul_16x16x5_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan layout(#pipeline_layout_1) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x5xf32>>
          %1 = hal.interface.binding.subspan layout(#pipeline_layout_1) set(0) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<5x16xf32>>
          %2 = hal.interface.binding.subspan layout(#pipeline_layout_1) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<16x16xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [16, 5], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16x5xf32>> -> tensor<16x5xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [5, 16], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<5x16xf32>> -> tensor<5x16xf32>
          %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<16x16xf32>> -> tensor<16x16xf32>
          %6 = linalg.matmul ins(%3, %4 : tensor<16x5xf32>, tensor<5x16xf32>) outs(%5 : tensor<16x16xf32>) -> tensor<16x16xf32>
          flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : tensor<16x16xf32> -> !flow.dispatch.tensor<readwrite:tensor<16x16xf32>>
          return
        }
      }
    }
  }
  hal.executable private @example_module_dispatch_2 {
    hal.executable.variant public @vulkan_spirv_fb target(<"vulkan-spirv", "vulkan-spirv-fb", {iree.gpu.target = #target}>) {
      hal.executable.export public @example_module_dispatch_2_generic_16x16_f32 ordinal(0) layout(#pipeline_layout_2) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @example_module_dispatch_2_generic_16x16_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan layout(#pipeline_layout_2) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x16xf32>>
          %1 = hal.interface.binding.subspan layout(#pipeline_layout_2) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16x16xf32>> -> tensor<16x16xf32>
          %3 = tensor.empty() : tensor<16xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<16x16xf32>) outs(%3 : tensor<16xf32>) {
          ^bb0(%in: f32, %out: f32):
            %5 = arith.addf %out, %in : f32
            linalg.yield %5 : f32
          } -> tensor<16xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0], sizes = [16], strides = [1] : tensor<16xf32> -> !flow.dispatch.tensor<writeonly:tensor<16xf32>>
          return
        }
      }
    }
  }
}

/// We test first with threading off so that the printers are legible.
// RUN: iree-compile %s --iree-hal-target-backends=vulkan \
// RUN:   --iree-codegen-transform-dialect-library=%p/transform_library.mlir@kernel_config \
// RUN:   --compile-from=executable-sources \
// RUN:   --compile-to=executable-targets \
// RUN:   --mlir-disable-threading | \
// RUN: FileCheck %s --check-prefixes=CODEGEN-PRINTER

// CODEGEN-PRINTER:     IR printer: Setting matmul strategy to custom_transform_strategy
// CODEGEN-PRINTER:       translation_info = #iree_codegen.translation_info<TransformDialectCodegen codegen_spec = @custom_transform_strategy>
// CODEGEN-PRINTER:     IR printer: Setting reduce strategy to base vectorize top-level
// CODEGEN-PRINTER:       translation_info = #iree_codegen.translation_info<SPIRVBaseVectorize workgroup_size = [16, 1, 1]>

/// Then test with threading to make sure it runs
// RUN: iree-compile %s --iree-hal-target-backends=vulkan \
// RUN:   --iree-codegen-transform-dialect-library=%p/transform_library.mlir@kernel_config \
// RUN:   --compile-from=executable-sources \
// RUN:   --compile-to=executable-targets \
// RUN:   --mlir-disable-threading | \
// RUN: FileCheck %s --check-prefixes=CODEGEN

// CODEGEN: spirv.func @example_module_dispatch_0_generic_80_f32
// CODEGEN: spirv.func @example_module_dispatch_1_matmul_16x16x5_f32
// CODEGEN: spirv.func @example_module_dispatch_2_generic_16x16_f32
