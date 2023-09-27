// RUN: iree-opt --split-input-file %s --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target)))" --iree-codegen-llvmcpu-use-transform-dialect=%p/transform_dialect_dummy_spec.mlir | FileCheck %s
// RUN: iree-opt --split-input-file %s --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target)))" --iree-codegen-transform-library-file-name=%p/transform_dialect_dummy_spec.mlir | FileCheck %s --check-prefix=CONFIG

// If we set the config on the command line, it takes precedence.
// CHECK: IR printer: from_flag

// When we include the library, we should honor the config we set in the
// attribute.
// CONFIG: IR printer: from_config

#blank_config = #iree_codegen.lowering_config<tile_sizes = []>
#translation = #iree_codegen.translation_info<TransformDialectCodegen codegen_spec=@print_config>
#config = #iree_codegen.compilation_info<lowering_config = #blank_config, translation_info = #translation, workgroup_size = []>

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "broadwell", cpu_features = "+cmov,+mmx,+popcnt,+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+avx,+avx2,+fma,+bmi,+bmi2,+pclmul,+adx,+cx16,+cx8,+crc32,+f16c,+fsgsbase,+fxsr,+invpcid,+lzcnt,+movbe,+prfchw,+rdrnd,+rdseed,+sahf,+x87,+xsave,+xsaveopt", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 32 : index, target_triple = "x86_64-unknown-unknown-eabi-elf", ukernels = false}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>

#device_target_llvm_cpu = #hal.device.target<"llvm-cpu", {executable_targets = [#executable_target_embedded_elf_x86_64_]}>
module attributes {hal.device.targets = [#device_target_llvm_cpu]} {
  hal.executable private @matmul_4x2304x768_f32_dispatch_0 {
    hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ attributes {transform.target_tag = "payload_root"} {
      hal.executable.export public @matmul_4x2304x768_f32_dispatch_0_generic_4x2304x768_f32 ordinal(0) layout(#pipeline_layout) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice
        hal.return %x, %y, %z : index, index, index
      }

      builtin.module {
        func.func @matmul_4x2304x768_f32_dispatch_0_generic_4x2304x768_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4x768xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<768x2304xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<4x2304xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [4, 768], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4x768xf32>> -> tensor<4x768xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [768, 2304], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<768x2304xf32>> -> tensor<768x2304xf32>
          %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [4, 2304], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<4x2304xf32>> -> tensor<4x2304xf32>
          %6 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<4x768xf32>, tensor<768x2304xf32>) outs(%5 : tensor<4x2304xf32>)
            attrs =  {compilation_info = #config} {
          ^bb0(%in: f32, %in_0: f32, %out: f32):
            %7 = arith.mulf %in, %in_0 fastmath<fast> : f32
            %8 = arith.addf %out, %7 fastmath<fast> : f32
            linalg.yield %8 : f32
          } -> tensor<4x2304xf32>
          flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [4, 2304], strides = [1, 1] : tensor<4x2304xf32> -> !flow.dispatch.tensor<readwrite:tensor<4x2304xf32>>
          return
        }
      }
    }
  }
}
