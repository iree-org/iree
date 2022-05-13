// RUN: iree-opt --pass-pipeline='hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target))' --split-input-file %s | FileCheck %s

// Check that this dispatch compiles to vectors and that there are no allocas.
// By proxy checks that destination passing style kicked in correctly
// and no CSE was run between first level tile + fuse + distribute
// and the conversion to destination passing style. Running CSE
// before hoists the fill and the init_tensor out of the loop causing
// issues with the conversion.
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0, d1) -> (d0)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm", "embedded-elf-x86_64", {
  cpu_features = "",
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-unknown-unknown-eabi-elf"}>
#executable_layout5 = #hal.executable.layout<push_constants = 2, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>]
  >]>
hal.executable private @check_no_cse {
  hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.entry_point public @check_no_cse ordinal(0) layout(#executable_layout5)
    builtin.module {
      func.func @check_no_cse() {
        %cst = arith.constant 3.840000e+02 : f32
        %cst_0 = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = arith.index_cast %0 {stream.alignment = 512 : index, stream.values = [0 : index, 10752 : index]} : i32 to index
        %3 = arith.index_cast %1 {stream.alignment = 512 : index, stream.values = [10752 : index, 21504 : index]} : i32 to index
        %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%2) alignment(64) : !flow.dispatch.tensor<readonly:7x384xf32>
        %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%3) alignment(64) : !flow.dispatch.tensor<writeonly:7xf32>
        %6 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [7, 384], strides = [1, 1] : !flow.dispatch.tensor<readonly:7x384xf32> -> tensor<7x384xf32>
        %7 = linalg.init_tensor [7] : tensor<7xf32>
        %8 = linalg.fill ins(%cst_0 : f32) outs(%7 : tensor<7xf32>) -> tensor<7xf32>
        %9 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "reduction"]} ins(%6 : tensor<7x384xf32>) outs(%8 : tensor<7xf32>) {
        ^bb0(%arg0: f32, %arg1: f32):
          %11 = arith.addf %arg1, %arg0 : f32
          linalg.yield %11 : f32
        } -> tensor<7xf32>
        %10 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%9 : tensor<7xf32>) outs(%7 : tensor<7xf32>) {
        ^bb0(%arg0: f32, %arg1: f32):
          %11 = arith.divf %arg0, %cst : f32
          linalg.yield %11 : f32
        } -> tensor<7xf32>
        flow.dispatch.tensor.store %10, %5, offsets = [0], sizes = [7], strides = [1] : tensor<7xf32> -> !flow.dispatch.tensor<writeonly:7xf32>
        return
      }
    }
  }
}
//      CHECK: func.func @check_no_cse()
//  CHECK-NOT:    memref.alloc
//      CHECK:    %[[FOR:.+]] = scf.for
//      CHECK:    %[[DIVF:.+]] = arith.divf %[[FOR]]
//      CHECK:    %[[RES:.+]] = vector.extract %[[DIVF]]
//      CHECK:    memref.store %[[RES]]

// -----

// Checks that the ops are padded and vectorized. The test sets tiling sizes to
// be non-divisible by problem sizes. If padding and vectorizing are kicked in,
// vector ops will be generated.
#compilation = #iree_codegen.compilation_info<
    lowering_config = <tile_sizes = [[65, 65, 0], [8, 32, 0], [0, 0, 16]]>,
    translation_info  = <CPUDoubleTilingPadExpert>,
    workgroup_size = []>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @preset_config_matmul  {
  hal.executable.variant @system_elf_x86_64, target = <"llvm", "system-elf-x86_64"> {
    hal.executable.entry_point @preset_config_matmul layout(#executable_layout)
    builtin.module {
      func.func @preset_config_matmul() {
        %cst = arith.constant 0.000000e+00 : f32
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:128x49xf32>
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:49x512xf32>
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:128x512xf32>
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [128, 49], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:128x49xf32> -> tensor<128x49xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [49, 512], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:49x512xf32> -> tensor<49x512xf32>
        %init = linalg.init_tensor [128, 512] : tensor<128x512xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<128x512xf32>) -> tensor<128x512xf32>
        %gemm = linalg.matmul {compilation_info = #compilation}
            ins(%lhs, %rhs : tensor<128x49xf32>, tensor<49x512xf32>)
            outs(%fill : tensor<128x512xf32>) -> tensor<128x512xf32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [128, 512], strides = [1, 1]
            : tensor<128x512xf32> -> !flow.dispatch.tensor<writeonly:128x512xf32>
        return
      }
    }
  }
}
// CHECK: func.func @preset_config_matmul
// CHECK:   vector.outerproduct

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm", "embedded-elf-x86_64", {
  cpu_features = "",
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-unknown-unknown-eabi-elf"}>
#executable_layout = #hal.executable.layout<push_constants = 2, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>]
  >]>
hal.executable private @check_buffer_ops_vectorization {
  hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.entry_point public @check_buffer_ops_vectorization ordinal(0) layout(#executable_layout)
    builtin.module {
      func.func @check_buffer_ops_vectorization() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : memref<128x1024xi32>
        memref.assume_alignment %0, 64 : memref<128x1024xi32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : memref<128x1536xi32>
        memref.assume_alignment %1, 64 : memref<128x1536xi32>
        %2 = memref.subview %1[0, 0] [128, 1024] [1, 1] : memref<128x1536xi32> to memref<128x1024xi32, affine_map<(d0, d1) -> (d0 * 1536 + d1)>>
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
          ins(%0 : memref<128x1024xi32>)
          outs(%2 : memref<128x1024xi32, affine_map<(d0, d1) -> (d0 * 1536 + d1)>>) {
        ^bb0(%arg0: i32, %arg1: i32):
          linalg.yield %arg0 : i32
        }
        return
      }
    }
  }
}
// CHECK:      #{{.+}} = #iree_codegen.translation_info<CPUBufferOpsTileAndVectorize
// CHECK:      func.func @check_buffer_ops_vectorization
// CHECK:        vector.load
// CHECK-NEXT:   vector.store
