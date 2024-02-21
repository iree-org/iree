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

#target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader, GroupNonUniform], [SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, api=Vulkan, #spirv.resource_limits<max_compute_workgroup_size = [128, 128, 64], subgroup_size = 64, cooperative_matrix_properties_khr = []>>

module attributes {hal.device.targets = [#hal.device.target<"vulkan", {executable_targets = [#hal.executable.target<"vulkan", "vulkan-spirv-fb", {spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader, GroupNonUniform], [SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, api=Vulkan, #spirv.resource_limits<max_compute_workgroup_size = [128, 128, 64], subgroup_size = 64, cooperative_matrix_properties_khr = []>>}>]}>]} {
  hal.executable private @example_module_dispatch_0 {
    hal.executable.variant public @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {spirv.target_env = #target_env}>) {
      hal.executable.export public @example_module_dispatch_0_generic_80_f32 ordinal(0) layout(
                                   #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @example_module_dispatch_0_generic_80_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<80xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<80xf32>>
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
    hal.executable.variant public @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {spirv.target_env = #target_env}>) {
      hal.executable.export public @example_module_dispatch_1_matmul_16x16x5_f32 ordinal(0) layout(
                                   #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @example_module_dispatch_1_matmul_16x16x5_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x5xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<5x16xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<16x16xf32>>
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
    hal.executable.variant public @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {spirv.target_env = #target_env}>) {
      hal.executable.export public @example_module_dispatch_2_generic_16x16_f32 ordinal(0) layout(
                                   #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @example_module_dispatch_2_generic_16x16_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x16xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16xf32>>
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
// CODEGEN-PRINTER:       translation_info = #iree_codegen.translation_info<SPIRVBaseVectorize>, workgroup_size = [16 : index, 1 : index, 1 : index]

/// Then test with threading to make sure it runs
// RUN: iree-compile %s --iree-hal-target-backends=vulkan \
// RUN:   --iree-codegen-transform-dialect-library=%p/transform_library.mlir@kernel_config \
// RUN:   --compile-from=executable-sources \
// RUN:   --compile-to=executable-targets \
// RUN:   --mlir-disable-threading | \
// RUN: FileCheck %s --check-prefixes=CODEGEN

// CODEGEN: Ran custom_transform_strategy
// CODEGEN: spirv.func @example_module_dispatch_0_generic_80_f32
// CODEGEN: hal.executable private @example_module_dispatch_1
// CODEGEN:   #iree_codegen.translation_info<TransformDialectCodegen codegen_spec = @custom_transform_strategy>
// CODEGEN:     spirv.func @example_module_dispatch_1_matmul_16x16x5_f32
// CODEGEN: spirv.func @example_module_dispatch_2_generic_16x16_f32
