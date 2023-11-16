// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-select-lowering-strategy)))' %s | FileCheck %s

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
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
  hal.executable.variant public @embedded_elf_arm_64 target(#executable_target_embedded_elf_arm_64_) {
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

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[8, 8, 0, 0, 0, 0], [1, 1, 0, 4, 4, 0], [0, 0, 1, 0, 0, 1]{{\]}}
//      CHECK: func.func @mmt4d_384x384x512_4x1x4_dispatch_0()
//      CHECK:   linalg.mmt4d
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "cascadelake", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 32 : index, target_triple = "x86_64-unknown-unknown-eabi-elf", ukernels = true}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 28, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer, ReadOnly>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @batch_mmt4d {
  hal.executable.variant public @embedded_elf_x86_64 target(#executable_target_embedded_elf_x86_64_) {
    hal.executable.export public @batch_mmt4d ordinal(0) layout(#pipeline_layout)
    builtin.module {
      func.func @batch_mmt4d() {
        %c32_i64 = arith.constant 32 : i64
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = hal.interface.constant.load[3] : i32
        %4 = arith.extui %0 : i32 to i64
        %5 = arith.extui %1 : i32 to i64
        %6 = arith.shli %5, %c32_i64 : i64
        %7 = arith.ori %4, %6 : i64
        %8 = arith.index_castui %7 {stream.alignment = 64 : index} : i64 to index
        %9 = arith.extui %2 : i32 to i64
        %10 = arith.extui %3 : i32 to i64
        %11 = arith.shli %10, %c32_i64 : i64
        %12 = arith.ori %9, %11 : i64
        %13 = arith.index_castui %12 : i64 to index
        %14 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x10x32x8x1xf32>>
        %15 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%8) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x80x32x4x1xf32>>
        %16 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%13) : !flow.dispatch.tensor<writeonly:tensor<128x10x80x8x4xf32>>
        %17 = flow.dispatch.tensor.load %14, offsets = [0, 0, 0, 0, 0], sizes = [128, 10, 32, 8, 1], strides = [1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x10x32x8x1xf32>> -> tensor<128x10x32x8x1xf32>
        %18 = flow.dispatch.tensor.load %15, offsets = [0, 0, 0, 0, 0], sizes = [128, 80, 32, 4, 1], strides = [1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x80x32x4x1xf32>> -> tensor<128x80x32x4x1xf32>
        %19 = tensor.empty() : tensor<128x10x80x8x4xf32>
        %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<128x10x80x8x4xf32>) -> tensor<128x10x80x8x4xf32>
        %21 = linalg.batch_mmt4d ins(%17, %18 : tensor<128x10x32x8x1xf32>, tensor<128x80x32x4x1xf32>) outs(%20 : tensor<128x10x80x8x4xf32>) -> tensor<128x10x80x8x4xf32>
        flow.dispatch.tensor.store %21, %16, offsets = [0, 0, 0, 0, 0], sizes = [128, 10, 80, 8, 4], strides = [1, 1, 1, 1, 1] : tensor<128x10x80x8x4xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x10x80x8x4xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 10, 40, 0, 0, 0, 0], [1, 1, 1, 0, 8, 4, 0], [0, 0, 0, 1, 0, 0, 1]{{\]}}>
//      CHECK: func.func @batch_mmt4d()
//      CHECK:   linalg.batch_mmt4d
// CHECK-SAME:     lowering_config = #[[CONFIG]]
