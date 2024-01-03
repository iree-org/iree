// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-select-lowering-strategy, iree-llvmcpu-lower-executable-target)))' --split-input-file %s | FileCheck %s

// Check that this dispatch compiles to vectors and that there are no allocas.
// By proxy checks that destination passing style kicked in correctly
// and no CSE was run between first level tile + fuse + distribute
// and the conversion to destination passing style. Running CSE
// before hoists the fill and the empty out of the loop causing
// issues with the conversion.
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0, d1) -> (d0)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  cpu_features = "",
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-none-elf"}>
#pipeline_layout5 = #hal.pipeline.layout<push_constants = 2, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>]
  >]>
hal.executable private @check_no_cse {
  hal.executable.variant public @embedded_elf_x86_64 target(#executable_target_embedded_elf_x86_64_) {
    hal.executable.export public @check_no_cse ordinal(0) layout(#pipeline_layout5) {
    ^bb0(%arg0: !hal.device, %arg1: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @check_no_cse() {
        %cst = arith.constant 3.840000e+02 : f32
        %cst_0 = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = arith.index_cast %0 {stream.alignment = 512 : index, stream.values = [0 : index, 10752 : index]} : i32 to index
        %3 = arith.index_cast %1 {stream.alignment = 512 : index, stream.values = [10752 : index, 21504 : index]} : i32 to index
        %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%2) : !flow.dispatch.tensor<readonly:tensor<7x384xf32>>
        %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%3) : !flow.dispatch.tensor<writeonly:tensor<7xf32>>
        %6 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [7, 384], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<7x384xf32>> -> tensor<7x384xf32>
        %7 = tensor.empty() : tensor<7xf32>
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
        flow.dispatch.tensor.store %10, %5, offsets = [0], sizes = [7], strides = [1] : tensor<7xf32> -> !flow.dispatch.tensor<writeonly:tensor<7xf32>>
        return
      }
    }
  }
}
// CHECK-LABEL: func.func @check_no_cse()
//       CHECK:    scf.for
//       CHECK:      arith.addf
//       CHECK:    vector.reduction <add>
//       CHECK:    arith.divf
//       CHECK:    vector.maskedstore

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  cpu_features = "",
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 64 : index,
  target_triple = "x86_64-none-elf"}>
#pipeline_layout5 = #hal.pipeline.layout<push_constants = 2, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>]
  >]>
hal.executable private @peel_partially_unaligned_matmul {
  hal.executable.variant public @embedded_elf_x86_64 target(#executable_target_embedded_elf_x86_64_) {
    hal.executable.export public @peel_partially_unaligned_matmul ordinal(0) layout(#pipeline_layout5) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @peel_partially_unaligned_matmul() {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = hal.interface.constant.load[3] : i32
        %4 = arith.index_castui %0 {stream.alignment = 128 : index, stream.values = [0 : index, 131712 : index]} : i32 to index
        %5 = arith.index_castui %1 {stream.alignment = 64 : index, stream.values = [576704 : index, 1763072 : index]} : i32 to index
        %6 = arith.index_castui %2 {stream.alignment = 64 : index, stream.values = [908480 : index, 2094848 : index]} : i32 to index
        %7 = arith.index_castui %3 {stream.alignment = 128 : index, stream.values = [2304 : index, 134016 : index]} : i32 to index
        %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%4) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x576xf32>>
        %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%5) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<576x144xf32>>
        %10 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%6) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x144xf32>>
        %11 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%7) : !flow.dispatch.tensor<writeonly:tensor<1x144xf32>>
        %12 = flow.dispatch.tensor.load %8, offsets = [0, 0], sizes = [1, 576], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x576xf32>> -> tensor<1x576xf32>
        %13 = flow.dispatch.tensor.load %9, offsets = [0, 0], sizes = [576, 144], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<576x144xf32>> -> tensor<576x144xf32>
        %14 = flow.dispatch.tensor.load %10, offsets = [0, 0], sizes = [1, 144], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x144xf32>> -> tensor<1x144xf32>
        %15 = tensor.empty() : tensor<1x144xf32>
        %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<1x144xf32>) -> tensor<1x144xf32>
        %17 = linalg.matmul ins(%12, %13 : tensor<1x576xf32>, tensor<576x144xf32>) outs(%16 : tensor<1x144xf32>) -> tensor<1x144xf32>
        %18 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %14 : tensor<1x144xf32>, tensor<1x144xf32>) outs(%15 : tensor<1x144xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %19 = arith.addf %in, %in_0 : f32
          %20 = arith.maximumf %19, %cst : f32
          linalg.yield %20 : f32
        } -> tensor<1x144xf32>
        flow.dispatch.tensor.store %18, %11, offsets = [0, 0], sizes = [1, 144], strides = [1, 1] : tensor<1x144xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x144xf32>>
        return
      }
    }
  }
}
// Checks that the bounded stack allocation are created.
// CHECK-LABEL: func.func @peel_partially_unaligned_matmul
// Main loop:
//       CHECK:     vector.fma
//       CHECK:     arith.addf {{.*}} : vector<
//       CHECK:     arith.maximumf {{.*}} : vector<
//
// Peeled loop:
//       CHECK:     vector.fma
//       CHECK:     arith.addf {{.*}} : vector<
//       CHECK:     arith.maximumf {{.*}} : vector<

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  cpu_features = "",
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-none-elf"}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 6, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>]
  >]>
hal.executable private @batch_matmul_dynamic {
  hal.executable.variant public @embedded_elf_x86_64 target(#executable_target_embedded_elf_x86_64_) {
    hal.executable.export public @batch_matmul_dynamic ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index, %arg4 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @batch_matmul_dynamic() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = hal.interface.constant.load[3] : i32
        %4 = hal.interface.constant.load[4] : i32
        %5 = hal.interface.constant.load[5] : i32
        %6 = arith.index_cast %0 : i32 to index
        %7 = arith.index_cast %1 : i32 to index
        %8 = arith.index_cast %2 : i32 to index
        %9 = arith.index_cast %3 : i32 to index
        %10 = arith.index_cast %4 : i32 to index
        %11 = arith.index_cast %5 : i32 to index
        %12 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%6, %7, %9}
        %13 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%10, %11, %8}
        %14 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?x?x?xf32>>{%6, %7, %8}
        %15 = flow.dispatch.tensor.load %12, offsets = [0, 0, 0], sizes = [%6, %7, %9], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%6, %7, %9} -> tensor<?x?x?xf32>
        %16 = flow.dispatch.tensor.load %13, offsets = [0, 0, 0], sizes = [%10, %11, %8], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%10, %11, %8} -> tensor<?x?x?xf32>
        %17 = tensor.empty(%6, %7, %8) : tensor<?x?x?xf32>
        %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
        %19 = linalg.batch_matmul ins(%15, %16 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%18 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
        flow.dispatch.tensor.store %19, %14, offsets = [0, 0, 0], sizes = [%6, %7, %8], strides = [1, 1, 1] : tensor<?x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x?xf32>>{%6, %7, %8}
        return
      }
    }
  }
}
// CHECK-LABEL: func.func @batch_matmul_dynamic
//       CHECK:   vector.fma

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  cpu_features = "",
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "x86_64-none-elf"}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 2, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>]
  >]>
hal.executable private @check_buffer_ops_vectorization {
  hal.executable.variant public @embedded_elf_x86_64 target(#executable_target_embedded_elf_x86_64_) {
    hal.executable.export public @check_buffer_ops_vectorization ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @check_buffer_ops_vectorization() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<128x1024xi32>
        memref.assume_alignment %0, 64 : memref<128x1024xi32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<128x1536xi32>
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
// CHECK-LABEL:  #{{.+}} = #iree_codegen.translation_info<CPUBufferOpsTileAndVectorize
//       CHECK:      func.func @check_buffer_ops_vectorization
//       CHECK:        vector.load
//       CHECK:        vector.store

// -----

hal.executable private @vectorize_fill_conv2d_generic {
  hal.executable.variant public @embedded_elf_x86_64 target(#hal.executable.target<
    "llvm-cpu", "embedded-elf-x86_64", {
      cpu_features = "",
      data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
      native_vector_size = 16 : index,
      target_triple = "x86_64-none-elf"
    }>) {
    hal.executable.export public @vectorize_fill_conv2d_generic ordinal(0) layout(
      #hal.pipeline.layout<push_constants = 0, sets = [
        #hal.descriptor_set.layout<0, bindings = [
          #hal.descriptor_set.binding<0, storage_buffer>,
          #hal.descriptor_set.binding<1, storage_buffer>,
          #hal.descriptor_set.binding<2, storage_buffer>]
        >]>
      ) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index, %arg4 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @vectorize_fill_conv2d_generic() {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 3.000000e+00 : f32
        %cst_1 = arith.constant 6.000000e+00 : f32
        %cst_2 = arith.constant 0.166666672 : f32
        %cst_3 = arith.constant dense<0.0> : tensor<16xf32>
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x225x225x3xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<3x3x3x16xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 225, 225, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x225x225x3xf32>> -> tensor<1x225x225x3xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 3, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x3x16xf32>> -> tensor<3x3x3x16xf32>
        %5 = tensor.empty() : tensor<1x112x112x16xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>
        %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x225x225x3xf32>, tensor<3x3x3x16xf32>) outs(%6 : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>
        %8 = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
            iterator_types = ["parallel", "parallel", "parallel", "parallel"
          ]} ins(%cst_3, %7 : tensor<16xf32>, tensor<1x112x112x16xf32>) outs(%5 : tensor<1x112x112x16xf32>) {
        ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
          %9 = arith.addf %arg0, %arg1 : f32
          %10 = arith.addf %9, %cst_0 : f32
          %11 = arith.cmpf olt, %10, %cst : f32
          %12 = arith.select %11, %cst, %10 : f32
          %13 = arith.cmpf olt, %cst_1, %10 : f32
          %14 = arith.select %13, %cst_1, %12 : f32
          %15 = arith.mulf %9, %14 : f32
          %16 = arith.mulf %15, %cst_2 : f32
          linalg.yield %16 : f32
        } -> tensor<1x112x112x16xf32>
        flow.dispatch.tensor.store %8, %2, offsets = [0, 0, 0, 0], sizes = [1, 112, 112, 16], strides = [1, 1, 1, 1] : tensor<1x112x112x16xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL:  func.func @vectorize_fill_conv2d_generic
//   CHECK-NOT:    memref.alloca
//   CHECK-NOT:    linalg.fill
//       CHECK:    vector.fma
//   CHECK-NOT:    linalg.generic
//       CHECK:    arith.cmpf olt, %{{.+}}, %{{.+}} : vector<4x4xf32>

// -----

hal.executable private @multi_result {
  hal.executable.variant public @embedded_elf_x86_64 target(<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>) {
    hal.executable.export public @multi_result ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @multi_result() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 1.000000e-03 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x128xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x256xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256xf32>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<writeonly:tensor<128x256xf32>>
        %5 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x256xf32>>
        %6 = hal.interface.binding.subspan set(0) binding(5) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x256xf32>>
        %7 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [64, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x128xf32>> -> tensor<64x128xf32>
        %8 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x256xf32>> -> tensor<128x256xf32>
        %9 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [256], strides = [1] : !flow.dispatch.tensor<readonly:tensor<256xf32>> -> tensor<256xf32>
        %13 = tensor.empty() : tensor<64x256xf32>
        %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<64x256xf32>) -> tensor<64x256xf32>
        %15 = linalg.matmul ins(%7, %8 : tensor<64x128xf32>, tensor<128x256xf32>) outs(%14 : tensor<64x256xf32>) -> tensor<64x256xf32>
        %16 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel"]}
          ins(%15, %9 : tensor<64x256xf32>, tensor<256xf32>) outs(%13 : tensor<64x256xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %17 = arith.addf %in, %in_1 : f32
          linalg.yield %17 : f32
        } -> tensor<64x256xf32>
        flow.dispatch.tensor.store %15, %5, offsets = [0, 0], sizes = [64, 256], strides = [1, 1] : tensor<64x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x256xf32>>
        flow.dispatch.tensor.store %16, %6, offsets = [0, 0], sizes = [64, 256], strides = [1, 1] : tensor<64x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x256xf32>>
        return
      }
    }
  }
}
//    CHECK-LABEL: func @multi_result
//          CHECK:   scf.for
//          CHECK:     scf.for
//          CHECK:       scf.for
// CHECK-COUNT-16:         vector.fma
//          CHECK:       arith.addf %{{.+}}, %{{.+}} : vector<8x32xf32>

// -----

hal.executable private @quant_matmul_fusion {
  hal.executable.variant public @embedded_elf_x86_64 target(<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>) {
    hal.executable.export public @quant_matmul_fusion ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer, ReadOnly>, <4, storage_buffer, ReadOnly>, <5, storage_buffer, ReadOnly>, <6, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @quant_matmul_fusion() {
        %c0 = arith.constant 0 : index
        %c12_i32 = arith.constant 12 : i32
        %c-128_i32 = arith.constant -128 : i32
        %c127_i32 = arith.constant 127 : i32
        %c0_i32 = arith.constant 0 : i32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2304x24xi8>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<24x144xi8>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<144xi32>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<144xi32>>
        %4 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<144xi32>>
        %5 = hal.interface.binding.subspan set(0) binding(5) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<144xi8>>
        %6 = hal.interface.binding.subspan set(0) binding(6) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2304x144xi8>>
        %7 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2304, 24], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2304x24xi8>> -> tensor<2304x24xi8>
        %8 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [24, 144], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<24x144xi8>> -> tensor<24x144xi8>
        %9 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [144], strides = [1] : !flow.dispatch.tensor<readonly:tensor<144xi32>> -> tensor<144xi32>
        %10 = flow.dispatch.tensor.load %3, offsets = [0], sizes = [144], strides = [1] : !flow.dispatch.tensor<readonly:tensor<144xi32>> -> tensor<144xi32>
        %11 = flow.dispatch.tensor.load %4, offsets = [0], sizes = [144], strides = [1] : !flow.dispatch.tensor<readonly:tensor<144xi32>> -> tensor<144xi32>
        %12 = flow.dispatch.tensor.load %5, offsets = [0], sizes = [144], strides = [1] : !flow.dispatch.tensor<readonly:tensor<144xi8>> -> tensor<144xi8>
        %13 = tensor.empty() : tensor<2304x144xi8>
        %14 = tensor.empty() : tensor<2304x144xi32>
        %15 = linalg.fill ins(%c0_i32 : i32) outs(%14 : tensor<2304x144xi32>) -> tensor<2304x144xi32>
        %16 = linalg.matmul ins(%7, %8 : tensor<2304x24xi8>, tensor<24x144xi8>) outs(%15 : tensor<2304x144xi32>) -> tensor<2304x144xi32>
        %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9, %16, %10, %11, %12 : tensor<144xi32>, tensor<2304x144xi32>, tensor<144xi32>, tensor<144xi32>, tensor<144xi8>) outs(%13 : tensor<2304x144xi8>) {
        ^bb0(%in: i32, %in_0: i32, %in_1: i32, %in_2: i32, %in_3: i8, %out: i8):
          %18 = arith.muli %in_1, %c12_i32 : i32
          %19 = arith.subi %in_0, %18 : i32
          %20 = arith.addi %in, %19 : i32
          %21 = tosa.apply_scale %20, %in_2, %in_3 {double_round = true} : (i32, i32, i8) -> i32
          %22 = arith.addi %21, %c-128_i32 : i32
          %23 = arith.cmpi slt, %22, %c-128_i32 : i32
          %24 = arith.select %23, %c-128_i32, %22 : i32
          %25 = arith.cmpi sgt, %22, %c127_i32 : i32
          %26 = arith.select %25, %c127_i32, %24 : i32
          %27 = arith.trunci %26 : i32 to i8
          linalg.yield %27 : i8
        } -> tensor<2304x144xi8>
        flow.dispatch.tensor.store %17, %6, offsets = [0, 0], sizes = [2304, 144], strides = [1, 1] : tensor<2304x144xi8> -> !flow.dispatch.tensor<writeonly:tensor<2304x144xi8>>
        return
      }
    }
  }
}
//    CHECK-LABEL: func @quant_matmul_fusion
//          CHECK:   scf.for
//          CHECK:     scf.for
//          CHECK:       scf.for
//          CHECK:         arith.muli
//     CHECK-NEXT:         arith.addi
//          CHECK:         arith.muli
//     CHECK-NEXT:         arith.addi
//          CHECK:         arith.muli
//     CHECK-NEXT:         arith.addi
//          CHECK:         arith.muli
//     CHECK-NEXT:         arith.addi
//          CHECK:         arith.muli
//     CHECK-NEXT:         arith.addi
//          CHECK:         arith.muli
//     CHECK-NEXT:         arith.addi
//          CHECK:         arith.muli
//     CHECK-NEXT:         arith.addi
//          CHECK:         arith.muli
//     CHECK-NEXT:         arith.addi
//          CHECK:       %{{.+}} = tosa.apply_scale {{.+}} {double_round = true} : (vector<8x32xi32>, vector<8x32xi32>, vector<8x32xi8>) -> vector<8x32xi32>

// -----

hal.executable private @mmt4d_ukernel {
  hal.executable.variant public @embedded_elf_x86_64 target(<"llvm-cpu", "embedded-elf-x86_64",
      {cpu = "generic", cpu_features = "",
       data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
       native_vector_size = 16 : index, target_triple = "x86_64-none-elf",
       ukernels = "mmt4d"}>) {
    hal.executable.export public @ukernel_dispatch ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4, %arg5, %arg6
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @ukernel_dispatch() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x4x8x32xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x4x16x32xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<2x16x8x16xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 4, 8, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x4x8x32xf32>> -> tensor<2x4x8x32xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [16, 4, 16, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x4x16x32xf32>> -> tensor<16x4x16x32xf32>
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [2, 16, 8, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<2x16x8x16xf32>> -> tensor<2x16x8x16xf32>
        %6 = linalg.mmt4d ins(%3, %4 : tensor<2x4x8x32xf32>, tensor<16x4x16x32xf32>) outs(%5 : tensor<2x16x8x16xf32>) -> tensor<2x16x8x16xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0, 0], sizes = [2, 16, 8, 16], strides = [1, 1, 1, 1] : tensor<2x16x8x16xf32> -> !flow.dispatch.tensor<readwrite:tensor<2x16x8x16xf32>>
        return
      }
    }
  }
}
// CHECK-LABEL: func @ukernel_dispatch()
// Checks scf.for for distribution loops.
//       CHECK:   scf.for
//       CHECK:     scf.for
//   CHECK-NOT:       scf.for
//       CHECK:   iree_codegen.ukernel.generic "iree_uk_mmt4d"

// -----

hal.executable private @ukernel_pass_through {
  hal.executable.variant public @embedded_elf_x86_64 target(<
    "llvm-cpu", "embedded-elf-x86_64", {
      cpu = "generic", cpu_features = "",
      data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
      native_vector_size = 16 : index, target_triple = "x86_64-none-elf",
      ukernels = "all"}>) {
    hal.executable.export public @dispatch ordinal(0) layout(#hal.pipeline.layout<
      push_constants = 2, sets = [
        <0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>,
                        <2, storage_buffer>]>]>)
        attributes {translation_info = #iree_codegen.translation_info<CPUDefault>} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      hal.return %arg1, %arg2, %arg2 : index, index, index
    }
    builtin.module {
      func.func @dispatch() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = arith.index_castui %0 : i32 to index
        %3 = arith.index_castui %1 : i32 to index
        %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%2}
        %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%3}
        %6 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<?xf32>>{%2}
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %7 = affine.min affine_map<()[s0, s1, s2] -> (s0 - s1 * (s0 ceildiv s2), s0 ceildiv s2)>()[%2, %workgroup_id_x, %workgroup_count_x]
        %8 = affine.apply affine_map<()[s0, s1, s2] -> (s0 * (s1 ceildiv s2))>()[%workgroup_id_x, %2, %workgroup_count_x]
        %9 = flow.dispatch.tensor.load %4, offsets = [%8], sizes = [%7], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%2} -> tensor<?xf32>
        %10 = flow.dispatch.tensor.load %5, offsets = [%8], sizes = [%7], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%3} -> tensor<?xf32>
        %11 = tensor.empty(%7) : tensor<?xf32>
        %12 = iree_codegen.ukernel.generic "simple_mul_workgroup" ins(%9, %10 : tensor<?xf32>, tensor<?xf32>) outs(%11 : tensor<?xf32>) (%7 : index) -> tensor<?xf32>
        flow.dispatch.tensor.store %12, %6, offsets = [%8], sizes = [%7], strides = [1] : tensor<?xf32> -> !flow.dispatch.tensor<readwrite:tensor<?xf32>>{%2}
        return
      }
    }
  }
}
// CHECK-LABEL: hal.executable private @ukernel_pass_through
//       CHECK:   hal.executable.export public @dispatch
//  CHECK-NEXT:       %[[ARG1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:       %[[ARG2:[a-zA-Z0-9]+]]: index
//       CHECK:     hal.return %[[ARG1]], %[[ARG2]], %[[ARG2]]
//       CHECK:   func @dispatch
//       CHECK:     %[[INPUT0:.+]] = hal.interface.binding.subspan set(0) binding(0)
//  CHECK-SAME:         memref<?xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:     %[[INPUT1:.+]] = hal.interface.binding.subspan set(0) binding(1)
//  CHECK-SAME:         memref<?xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:     %[[OUTPUT:.+]] = hal.interface.binding.subspan set(0) binding(2)
//  CHECK-SAME:         memref<?xf32, #hal.descriptor_type<storage_buffer>>
//   CHECK-DAG:     %[[OFFSET:.+]] = affine.apply
//   CHECK-DAG:     %[[SIZE:.+]] = affine.min
//   CHECK-DAG:     %[[SUBVIEW_OUTPUT:.+]] = memref.subview %[[OUTPUT]][%[[OFFSET]]] [%[[SIZE]]]
//   CHECK-DAG:     %[[SUBVIEW_INPUT0:.+]] = memref.subview %[[INPUT0]][%[[OFFSET]]] [%[[SIZE]]]
//   CHECK-DAG:     %[[SUBVIEW_INPUT1:.+]] = memref.subview %[[INPUT1]][%[[OFFSET]]] [%[[SIZE]]]
//       CHECK:     iree_codegen.ukernel.generic "simple_mul_workgroup"
//  CHECK-SAME:         ins(%[[SUBVIEW_INPUT0]], %[[SUBVIEW_INPUT1]]
//  CHECK-SAME:         outs(%[[SUBVIEW_OUTPUT]]

// -----

hal.executable private @unsupported_ukernel_fallback_to_vectorization {
  hal.executable.variant public @embedded_elf_x86_64 target(<
    "llvm-cpu", "embedded-elf-x86_64", {
      cpu = "generic",
      cpu_features = "+fma,+avx512f",
      data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128",
      native_vector_size = 64 : index,
      target_triple = "x86_64-unknown-unknown-eabi-elf", ukernels = "all"}>
    ) {
    hal.executable.export public @unsupported_ukernel_fallback_to_vectorization ordinal(0)
      layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>)
      attributes {translation_info = #iree_codegen.translation_info<Mmt4dTilingExpert>} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @unsupported_ukernel_fallback_to_vectorization() {
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c132096 = arith.constant 132096 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x256x1x1xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c1024) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4x256x128x1xi8>>
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c132096) : !flow.dispatch.tensor<writeonly:tensor<1x4x1x128xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 256, 1, 1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x256x1x1xf32>> -> tensor<1x256x1x1xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [4, 256, 128, 1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x256x128x1xi8>> -> tensor<4x256x128x1xi8>
        %5 = tensor.empty() : tensor<1x4x1x128xf32>
        %6 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 2, 0, 0], [1, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]>} ins(%cst : f32) outs(%5 : tensor<1x4x1x128xf32>) -> tensor<1x4x1x128xf32>
        %7 = linalg.mmt4d {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 2, 0, 0, 0, 0], [1, 1, 0, 1, 128, 0], [0, 0, 1, 0, 0, 1]]>} ins(%3, %4 : tensor<1x256x1x1xf32>, tensor<4x256x128x1xi8>) outs(%6 : tensor<1x4x1x128xf32>) -> tensor<1x4x1x128xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 4, 1, 128], strides = [1, 1, 1, 1] : tensor<1x4x1x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x4x1x128xf32>>
        return
      }
    }
  }
}
// CHECK-LABEL: func.func @unsupported_ukernel_fallback_to_vectorization
// CHECK:         vector.fma

// -----

// Check Armv9 Streaming SVE mode is enabled for the following pipelines:
//
//   * CPUBufferOpsTileAndVectorize
//   * CPUDoubleTilingPeelingExpert
//   * CPUConvTileAndDecomposeExpert

#pipeline_layout = #hal.pipeline.layout<push_constants = 2, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  cpu_features = "+sve,+sme",
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "aarch64-none-elf"
}>

hal.executable private @aarch64_ssve__cpu_buffer_ops_tile_and_vectorize {
  hal.executable.variant public @embedded_elf_arm_64 target(#executable_target_embedded_elf_arm_64_) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout) attributes {
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0], [1], [0], [0]]>,
      translation_info = #iree_codegen.translation_info<CPUBufferOpsTileAndVectorize>
    } {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      hal.return %arg1, %arg2, %arg2 : index, index, index
    }
    builtin.module {
      func.func @dispatch() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %cst_0 = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : i32
        %6 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<1xf32>>
        %7 = tensor.empty() : tensor<1xf32>
        %8 = linalg.fill ins(%cst_0 : f32) outs(%7 : tensor<1xf32>) -> tensor<1xf32>
        flow.dispatch.tensor.store %8, %6, offsets = [0], sizes = [1], strides = [1] : tensor<1xf32> -> !flow.dispatch.tensor<readwrite:tensor<1xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL: @aarch64_ssve__cpu_buffer_ops_tile_and_vectorize
// CHECK: func.func @dispatch() attributes {arm_locally_streaming}

hal.executable private @aarch64_ssve__cpu_double_tiling_peeling_expert {
  hal.executable.variant public @embedded_elf_arm_64 target(#executable_target_embedded_elf_arm_64_) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout) attributes {
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0], [1], [0], [0]]>,
      translation_info = #iree_codegen.translation_info<CPUDoubleTilingPeelingExpert>
    } {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      hal.return %arg1, %arg2, %arg2 : index, index, index
    }
    builtin.module {
      func.func @dispatch() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %cst_0 = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : i32
        %6 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<1xf32>>
        %7 = tensor.empty() : tensor<1xf32>
        %8 = linalg.fill ins(%cst_0 : f32) outs(%7 : tensor<1xf32>) -> tensor<1xf32>
        flow.dispatch.tensor.store %8, %6, offsets = [0], sizes = [1], strides = [1] : tensor<1xf32> -> !flow.dispatch.tensor<readwrite:tensor<1xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL: @aarch64_ssve__cpu_double_tiling_peeling_expert
// CHECK: func.func @dispatch() attributes {arm_locally_streaming}

hal.executable private @aarch64_ssve__cpu_conv_tile_and_decompose_expert {
  hal.executable.variant public @embedded_elf_arm_64 target(#executable_target_embedded_elf_arm_64_) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout) attributes {
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0], [1], [0], [0]]>,
      translation_info = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
    } {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      hal.return %arg1, %arg2, %arg2 : index, index, index
    }
    builtin.module {
      func.func @dispatch() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %cst_0 = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : i32
        %6 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<1xf32>>
        %7 = tensor.empty() : tensor<1xf32>
        %8 = linalg.fill ins(%cst_0 : f32) outs(%7 : tensor<1xf32>) -> tensor<1xf32>
        flow.dispatch.tensor.store %8, %6, offsets = [0], sizes = [1], strides = [1] : tensor<1xf32> -> !flow.dispatch.tensor<readwrite:tensor<1xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL: @aarch64_ssve__cpu_conv_tile_and_decompose_expert
// CHECK: func.func @dispatch() attributes {arm_locally_streaming}

// Check Armv9 Streaming SVE mode is not enabled if +sve and +sme are not
// specified.

#executable_target_embedded_elf_arm_64_no_sve = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  cpu_features = "+sme",
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "aarch64-none-elf"
}>

hal.executable private @aarch64_ssve_sve_disabled {
  hal.executable.variant public @embedded_elf_arm_64 target(#executable_target_embedded_elf_arm_64_no_sve) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout) attributes {
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0], [1], [0], [0]]>,
      translation_info = #iree_codegen.translation_info<CPUBufferOpsTileAndVectorize>
    } {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      hal.return %arg1, %arg2, %arg2 : index, index, index
    }
    builtin.module {
      func.func @dispatch() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %cst_0 = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : i32
        %6 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<1xf32>>
        %7 = tensor.empty() : tensor<1xf32>
        %8 = linalg.fill ins(%cst_0 : f32) outs(%7 : tensor<1xf32>) -> tensor<1xf32>
        flow.dispatch.tensor.store %8, %6, offsets = [0], sizes = [1], strides = [1] : tensor<1xf32> -> !flow.dispatch.tensor<readwrite:tensor<1xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL: @aarch64_ssve_sve_disabled
// CHECK-NOT: func.func @dispatch() attributes {arm_locally_streaming}
