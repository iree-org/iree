// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target)))' --split-input-file %s | FileCheck %s

hal.executable private @elem_pack {
  hal.executable.variant public @embedded_elf_x86_64, target = <"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-unknown-eabi-elf"}> {
    hal.executable.export public @elem_pack ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) attributes {translation_info = #iree_codegen.translation_info<CPUDataTiling>} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %c1 = arith.constant 1 : index
      %0 = affine.apply affine_map<()[s0] -> ((s0 ceildiv 8) ceildiv 64)>()[%arg1]
      %1 = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%arg2]
      hal.return %1, %0, %c1 : index, index, index
    }
    builtin.module {
      func.func @elem_pack() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<128x384xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x384x8x1xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 384], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x384xf32>> -> tensor<128x384xf32>
        %3 = tensor.empty() : tensor<128x384xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<128x384xf32>) outs(%3 : tensor<128x384xf32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.addf %in, %in : f32
          linalg.yield %7 : f32
        } -> tensor<128x384xf32>
        %5 = tensor.empty() : tensor<16x384x8x1xf32>
        %6 = tensor.pack %4 inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %5 : tensor<128x384xf32> -> tensor<16x384x8x1xf32>
        flow.dispatch.tensor.store %6, %1, offsets = [0, 0, 0, 0], sizes = [16, 384, 8, 1], strides = [1, 1, 1, 1] : tensor<16x384x8x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x384x8x1xf32>>
        return
      }
    }
  }
}
// CHECK: func.func @elem_pack
// CHECK:   %[[READ:.+]] = vector.transfer_read
// CHECK:   %[[ADD:.+]] = arith.addf %[[READ]], %[[READ]]
// CHECK:   vector.transfer_write %[[ADD]], %{{.+}}
