// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target)))' --split-input-file %s | FileCheck %s

hal.executable private @aligned_generic_pack {
  hal.executable.variant public @embedded_elf_x86_64, target = <"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}> {
    hal.executable.export public @aligned_generic_pack ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device):
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @aligned_generic_pack() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 3.40282347E+38 : f32
        %cst_0 = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<512xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<384x512xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<24x512x16x1xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [512], strides = [1] : !flow.dispatch.tensor<readonly:tensor<512xf32>> -> tensor<512xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [384, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<384x512xf32>> -> tensor<384x512xf32>
        %5 = tensor.empty() : tensor<24x512x16x1xf32>
        %6 = tensor.empty() : tensor<384x512xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3, %4 : tensor<512xf32>, tensor<384x512xf32>) outs(%6 : tensor<384x512xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %8 = arith.addf %in, %in_1 : f32
          %9 = arith.minimumf %8, %cst : f32
          %10 = arith.maximumf %9, %cst_0 : f32
          linalg.yield %10 : f32
        } -> tensor<384x512xf32>
        %pack = tensor.pack %7 inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %5 : tensor<384x512xf32> -> tensor<24x512x16x1xf32>
        flow.dispatch.tensor.store %pack, %2, offsets = [0, 0, 0, 0], sizes = [24, 512, 16, 1], strides = [1, 1, 1, 1] : tensor<24x512x16x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<24x512x16x1xf32>>
        return
      }
    }
  }
}
// CHECK-LABEL:     func.func @aligned_generic_pack
// CHECK:             %[[IN_0:.+]] = vector.broadcast %{{.+}} : vector<16xf32> to vector<16x16xf32>
// CHECK-COUNT-15:    %{{.+}} = vector.insert {{.+}} : vector<16xf32> into vector<16x16xf32>
// CHECK:             %[[IN_1:.+]] = vector.insert {{.+}} : vector<16xf32> into vector<16x16xf32>
// CHECK:             %[[T0:.+]] = arith.addf %[[IN_0]], %[[IN_1]] : vector<16x16xf32>
// CHECK:             %[[T1:.+]] = arith.minimumf %[[T0]], %{{.+}} : vector<16x16xf32>
// CHECK:             %[[T2:.+]] = arith.maximumf %[[T1]], %{{.+}} : vector<16x16xf32>
// CHECK-COUNT-16:    vector.extract %[[T2]]
// CHECK-COUNT-64:    vector.shuffle

// -----

hal.executable private @aligned_unpack_generic {
  hal.executable.variant public @embedded_elf_x86_64, target = <"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}> {
    hal.executable.export public @aligned_unpack_generic ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @aligned_unpack_generic() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 3.40282347E+38 : f32
        %cst_0 = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<24x32x16x16xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<512xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<384x512xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [24, 32, 16, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<24x32x16x16xf32>> -> tensor<24x32x16x16xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [512], strides = [1] : !flow.dispatch.tensor<readonly:tensor<512xf32>> -> tensor<512xf32>
        %5 = tensor.empty() : tensor<384x512xf32>
        %unpack = tensor.unpack %3 inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %5 : tensor<24x32x16x16xf32> -> tensor<384x512xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4, %unpack : tensor<512xf32>, tensor<384x512xf32>) outs(%5 : tensor<384x512xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %7 = arith.addf %in, %in_1 : f32
          %8 = arith.minimumf %7, %cst : f32
          %9 = arith.maximumf %8, %cst_0 : f32
          linalg.yield %9 : f32
        } -> tensor<384x512xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [384, 512], strides = [1, 1] : tensor<384x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<384x512xf32>>
        return
      }
    }
  }
}
// CHECK-LABEL:     func.func @aligned_unpack_generic
// CHECK:             %[[SRC:.+]] = hal.interface.binding.subspan {{.*}} : memref<24x32x16x16xf32, #hal.descriptor_type<storage_buffer>>
// CHECK:               %[[SUBVIEW:.+]] = memref.subview %{{.*}} memref<24x32x16x16xf32, #hal.descriptor_type<storage_buffer>> to memref<
// CHECK-COUNT-15:        vector.load %[[SUBVIEW]]
// CHECK:                 %[[LAST_LOAD:.+]] = vector.load %[[SUBVIEW]]
// CHECK-NEXT:            %[[IN_1:.+]] = vector.insert %[[LAST_LOAD]], %{{.*}}
// CHECK:                 %[[IN_0:.+]] = vector.broadcast %{{.+}} : vector<16xf32> to vector<16x16xf32>
// CHECK:                 %[[T0:.+]] = arith.addf %[[IN_0]], %[[IN_1]] : vector<16x16xf32>
// CHECK:                 %[[T1:.+]] = arith.minimumf %[[T0]], %{{.+}} : vector<16x16xf32>
// CHECK:                 %[[T2:.+]] = arith.maximumf %[[T1]], %{{.+}} : vector<16x16xf32>

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @unaligned_pack  {
  hal.executable.variant public @embedded_elf_x86_64, target = <"llvm-cpu", "embedded-elf-x86_64", {
    cpu_features = "+avx512f",
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 64 : index,
    target_triple = "x86_64-none-elf"
  }> {
  hal.executable.export public @unaligned_pack layout(#pipeline_layout) {
  ^bb0(%arg0: !hal.device):
    %c1 = arith.constant 1 : index
    hal.return %c1, %c1, %c1 : index, index, index
  }
    builtin.module {
      func.func @unaligned_pack() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<20x40xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x48x16x1xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [20, 40], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<20x40xf32>> -> tensor<20x40xf32>
        %3 = tensor.empty() : tensor<2x48x16x1xf32>
        %4 = tensor.pack %2 padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %3 : tensor<20x40xf32> -> tensor<2x48x16x1xf32>
        flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [2, 48, 16, 1], strides = [1, 1, 1, 1] : tensor<2x48x16x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x48x16x1xf32>>
        return
      }
    }
  }
}
// CHECK-LABEL:     func.func @unaligned_pack
// CHECK-COUNT-16:    vector.maskedload {{.+}} vector<16xf32>
// CHECK-COUNT-64:    vector.shuffle
