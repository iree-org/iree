// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target{test-lowering-configuration=true})))' %s | FileCheck %s

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map3 = affine_map<(d0)[s0] -> (s0, -d0 + 96)>
#map4 = affine_map<(d0)[s0] -> (s0, -d0 + 128)>
hal.executable private @mmt4d_384x384x512_4x1x4_dispatch_0 {
  hal.executable.variant public @embedded_elf_arm_64, target = #executable_target_embedded_elf_arm_64_ {
    hal.executable.export public @mmt4d_384x384x512_4x1x4_dispatch_0 layout(#pipeline_layout)
    builtin.module  {
      func.func @mmt4d_384x384x512_4x1x4_dispatch_0() {
        %c0 = arith.constant 0 : index
        %c96 = arith.constant 96 : index
        %c128 = arith.constant 128 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<96x384x4x1xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<128x384x4x1xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<96x128x4x4xf32>>
        %8 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [96, 384, 4, 1], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<96x384x4x1xf32>> -> tensor<96x384x4x1xf32>
        %10 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [128, 384, 4, 1], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<128x384x4x1xf32>> -> tensor<128x384x4x1xf32>
        %11 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [96, 384, 4, 4], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readwrite:tensor<96x128x4x4xf32>> -> tensor<96x128x4x4xf32>
        %12 = linalg.mmt4d {__internal_linalg_transform__ = "workgroup"}
            ins(%8, %10 : tensor<96x384x4x1xf32>, tensor<128x384x4x1xf32>)
            outs(%11 : tensor<96x128x4x4xf32>) -> tensor<96x128x4x4xf32>
        flow.dispatch.tensor.store %12, %2, offsets = [0, 0, 0, 0], sizes = [96, 128, 4, 4], strides = [1, 1, 1, 1]
            : tensor<96x128x4x4xf32> -> !flow.dispatch.tensor<readwrite:tensor<96x128x4x4xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32, 32, 0, 0, 0, 0], [1, 1, 0, 4, 4, 0], [0, 0, 1, 0, 0, 1]{{\]}}
//      CHECK: func.func @mmt4d_384x384x512_4x1x4_dispatch_0()
//      CHECK:   linalg.mmt4d
// CHECK-SAME:     lowering_config = #[[CONFIG]]
