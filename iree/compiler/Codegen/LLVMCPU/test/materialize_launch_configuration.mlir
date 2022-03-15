// RUN: iree-opt -pass-pipeline='hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target{test-lowering-configuration=true}))' -split-input-file %s | FileCheck %s

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors  {
  hal.executable.variant @llvm, target = <"llvm", "embedded-elf-arm_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-unknown-unknown-eabi-elf"
  }> {
    hal.executable.entry_point @matmul_tensors layout(#executable_layout)
    builtin.module {
      func @matmul_tensors() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %M = hal.interface.constant.load[0] : index
        %N = hal.interface.constant.load[1] : index
        %K = hal.interface.constant.load[2] : index
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:?x?xf32>{%M, %K}
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:?x?xf32>{%K, %N}
        %init_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:?x?xf32>{%M, %N}
        %result_binding = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:?x?xf32>{%M, %N}
              %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:?x?xf32>{%M, %K} -> tensor<?x?xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:?x?xf32>{%K, %N} -> tensor<?x?xf32>
        %init = flow.dispatch.tensor.load %init_binding, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:?x?xf32>{%M, %N} -> tensor<?x?xf32>
        %gemm = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
            : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>{%M, %N}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [16, 4, 64], [4, 4, 4]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUTileFuseAndVectorize>
//      CHECK: hal.executable.entry_point public @matmul_tensors
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @add {
  hal.executable.variant @llvm, target = <"llvm", "embedded-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }> {
    hal.executable.entry_point @add layout(#executable_layout)
    builtin.module {
      func @add() {
        %c0 = arith.constant 0 : index
        %dim0 = hal.interface.constant.load[0] : index
        %dim1 = hal.interface.constant.load[1] : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:?x?xf32>{%dim0, %dim1}
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:?xf32>{%dim1}
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:?x?xf32>{%dim0, %dim1}
        %3 = flow.dispatch.tensor.load %0, offsets=[0, 0], sizes=[%dim0, %dim1], strides=[1, 1] : !flow.dispatch.tensor<readonly:?x?xf32>{%dim0, %dim1} -> tensor<?x?xf32>
        %4 = flow.dispatch.tensor.load %1, offsets=[0], sizes=[%dim1], strides=[1] : !flow.dispatch.tensor<readonly:?xf32>{%dim1} -> tensor<?xf32>
        %5 = linalg.init_tensor [%dim0, %dim1] : tensor<?x?xf32>
        %6 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%3, %4 : tensor<?x?xf32>, tensor<?xf32>) outs(%5 : tensor<?x?xf32>) {
          ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
            %7 = arith.addf %arg0, %arg1 : f32
            linalg.yield %7 : f32
          } -> tensor<?x?xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%dim0, %dim1], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>{%dim0, %dim1}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64], [1, 4], [0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.entry_point public @add
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
hal.executable private @add4D  {
  hal.executable.variant @llvm, target = <"llvm", "embedded-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }> {
    hal.executable.entry_point @add4D layout(#executable_layout)
    builtin.module {
      func @add4D() {
        %d0 = hal.interface.constant.load[0] : index
        %d1 = hal.interface.constant.load[1] : index
        %d2 = hal.interface.constant.load[2] : index
        %d3 = hal.interface.constant.load[3] : index
        %arg1_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32)
            : !flow.dispatch.tensor<readonly:?x?x?x?xf32>{%d0, %d1, %d2, %d3}
        %arg2_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32)
            : !flow.dispatch.tensor<readonly:?x?x?x?xf32>{%d0, %d1, %d2, %d3}
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32)
            : !flow.dispatch.tensor<writeonly:?x?x?x?xf32>{%d0, %d1, %d2, %d3}
        %arg1 = flow.dispatch.tensor.load %arg1_binding, offsets = [0, 0, 0, 0], sizes = [%d0, %d1, %d2, %d3], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:?x?x?x?xf32>{%d0, %d1, %d2, %d3} -> tensor<?x?x?x?xf32>
        %arg2 = flow.dispatch.tensor.load %arg2_binding, offsets = [0, 0, 0, 0], sizes = [%d0, %d1, %d2, %d3], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:?x?x?x?xf32>{%d0, %d1, %d2, %d3} -> tensor<?x?x?x?xf32>
        %init = linalg.init_tensor [%d0, %d1, %d2, %d3] : tensor<?x?x?x?xf32>
        %add = linalg.generic {
            indexing_maps = [#map, #map, #map],
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
            ins(%arg1, %arg2 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%init : tensor<?x?x?x?xf32>) {
            ^bb0(%b0: f32, %b1: f32, %b2: f32):  // no predecessors
              %addf = arith.addf %b0, %b1 : f32
              linalg.yield %addf : f32
            } -> tensor<?x?x?x?xf32>
        flow.dispatch.tensor.store %add, %result_binding, offsets = [0, 0, 0, 0], sizes = [%d0, %d1, %d2, %d3], strides = [1, 1, 1, 1]
            : tensor<?x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?x?x?xf32>{%d0, %d1, %d2, %d3}
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 64, 64, 64], [1, 1, 1, 4], [0, 0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.entry_point public @add4D
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @batch_matmul_tensors {
  hal.executable.variant @llvm, target = <"llvm", "embedded-elf-arm_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-unknown-unknown-eabi-elf"
  }> {
    hal.executable.entry_point @batch_matmul_tensors layout(#executable_layout)
    builtin.module {
      func @batch_matmul_tensors() {
        %cst = arith.constant 0.000000e+00 : f32
        %B = hal.interface.constant.load[0] : index
        %M = hal.interface.constant.load[1] : index
        %N = hal.interface.constant.load[2] : index
        %K = hal.interface.constant.load[3] : index
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32)
            : !flow.dispatch.tensor<readonly:?x?x?xf32>{%B, %M, %K}
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32)
            : !flow.dispatch.tensor<readonly:?x?x?xf32>{%B, %K, %N}
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32)
            : !flow.dispatch.tensor<writeonly:?x?x?xf32>{%B, %M, %N}
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0, 0], sizes = [%B, %M, %K], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:?x?x?xf32>{%B, %M, %K} -> tensor<?x?x?xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0, 0], sizes = [%B, %K, %N], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:?x?x?xf32>{%B, %K, %N} -> tensor<?x?x?xf32>
        %init = linalg.init_tensor [%B, %M, %N] : tensor<?x?x?xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
        %batch_gemm = linalg.batch_matmul
            ins(%lhs, %rhs : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%fill : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
        flow.dispatch.tensor.store %batch_gemm, %result_binding, offsets = [0, 0, 0], sizes = [%B, %M, %N], strides = [1, 1, 1]
            : tensor<?x?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?x?xf32>{%B, %M, %N}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 64, 64, 0], [1, 16, 4, 64], [1, 4, 4, 4]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUTileFuseAndVectorize>
//      CHECK: hal.executable.entry_point public @batch_matmul_tensors
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:  linalg.batch_matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#compilation = #iree_codegen.compilation_info<
    lowering_config = <tile_sizes = [[64, 64, 0], [32, 32, 0], [0, 0, 32]]>,
    translation_info  = <CPUDoubleTilingExpert>,
    workgroup_size = []>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @preset_config_matmul_tensors  {
  hal.executable.variant @system_elf_x86_64, target = <"llvm", "system-elf-x86_64"> {
    hal.executable.entry_point @preset_config layout(#executable_layout)
    builtin.module {
      builtin.func @preset_config() {
        %cst = arith.constant 0.000000e+00 : f32
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:128x256xf32>
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:256x512xf32>
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:128x512xf32>
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [128, 256], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:128x256xf32> -> tensor<128x256xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [256, 512], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:256x512xf32> -> tensor<256x512xf32>
        %init = linalg.init_tensor [128, 512] : tensor<128x512xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<128x512xf32>) -> tensor<128x512xf32>
        %gemm = linalg.matmul {compilation_info = #compilation}
            ins(%lhs, %rhs : tensor<128x256xf32>, tensor<256x512xf32>)
            outs(%fill : tensor<128x512xf32>) -> tensor<128x512xf32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [128, 512], strides = [1, 1]
            : tensor<128x512xf32> -> !flow.dispatch.tensor<writeonly:128x512xf32>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [32, 32, 0], [0, 0, 32]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.entry_point
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: func @preset_config
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @tensor_insert {
  hal.executable.variant @system_elf_x86_64, target = <"llvm", "system-elf-x86_64"> {
    hal.executable.entry_point @tensor_insert_slice layout(#executable_layout)
    builtin.module {
      builtin.func @tensor_insert_slice() {
        %d0 = hal.interface.constant.load[0] : index
        %d1 = hal.interface.constant.load[1] : index
        %d2 = hal.interface.constant.load[2] : index
        %d3 = hal.interface.constant.load[3] : index
        %o0 = hal.interface.constant.load[4] : index
        %o1 = hal.interface.constant.load[5] : index
        %source_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:?x?xi32>{%d0, %d1}
        %dest_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readwrite:?x?xi32>{%d2, %d3}
        %source = flow.dispatch.tensor.load %source_binding, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:?x?xi32>{%d0, %d1} -> tensor<?x?xi32>
        %dest = flow.dispatch.tensor.load %dest_binding, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
            : !flow.dispatch.tensor<readwrite:?x?xi32>{%d2, %d3} -> tensor<?x?xi32>
        %result = tensor.insert_slice %source into %dest[%o0, %o1] [%d0, %d1] [1, 1] : tensor<?x?xi32> into tensor<?x?xi32>
        flow.dispatch.tensor.store %result, %dest_binding, offsets = [0, 0], sizes = [%d2, %d3], strides = [1, 1]
            : tensor<?x?xi32> -> !flow.dispatch.tensor<readwrite:?x?xi32>{%d2, %d3}
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDefault>
//      CHECK: hal.executable.entry_point public @tensor_insert_slice
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: tensor.insert_slice
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @extract_slice {
  hal.executable.variant @system_elf_x86_64, target = <"llvm", "system-elf-x86_64"> {
    hal.executable.entry_point @extract_slice layout(#executable_layout)
    builtin.module {
      builtin.func @extract_slice() {
        %d0 = hal.interface.constant.load[0] : index
        %d1 = hal.interface.constant.load[1] : index
        %d2 = hal.interface.constant.load[2] : index
        %d3 = hal.interface.constant.load[3] : index
        %o0 = hal.interface.constant.load[4] : index
        %o1 = hal.interface.constant.load[5] : index
        %source_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:?x?xi32>{%d0, %d1}
        %dest_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:?x?xi32>{%d2, %d3}
        %source = flow.dispatch.tensor.load %source_binding, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:?x?xi32>{%d0, %d1} -> tensor<?x?xi32>
        %dest = flow.dispatch.tensor.load %dest_binding, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
            : !flow.dispatch.tensor<writeonly:?x?xi32>{%d2, %d3} -> tensor<?x?xi32>
        %result = tensor.extract_slice %source[%o0, %o1] [%d0, %d1] [1, 1] : tensor<?x?xi32> to tensor<?x?xi32>
        flow.dispatch.tensor.store %result, %dest_binding, offsets = [0, 0], sizes = [%d2, %d3], strides = [1, 1]
            : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:?x?xi32>{%d2, %d3}
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDefault>
//      CHECK: hal.executable.entry_point public @extract_slice
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: tensor.extract_slice
// CHECK-SAME:     lowering_config = #[[CONFIG]]


// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @static_1d_fft_stage2  {
  hal.executable.variant @system_elf_x86_64, target = <"llvm", "system-elf-x86_64"> {
    hal.executable.entry_point @static_1d_fft_stage2 layout(#executable_layout)
    builtin.module {
      builtin.func @static_1d_fft_stage2() {
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %cst = arith.constant dense<[1.000000e+00, 6.12323426E-17]> : tensor<2xf32>
        %cst_0 = arith.constant dense<[-0.000000e+00, -1.000000e+00]> : tensor<2xf32>
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readwrite:32xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readwrite:32xf32>
        %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [32], strides = [1] : !flow.dispatch.tensor<readwrite:32xf32> -> tensor<32xf32>
        %3 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [32], strides = [1] : !flow.dispatch.tensor<readwrite:32xf32> -> tensor<32xf32>
        %4:2 = iree_linalg_ext.fft {__internal_linalg_transform__ = "workgroup"} ins(%c2, %cst, %cst_0 : index, tensor<2xf32>, tensor<2xf32>) outs(%2, %3 : tensor<32xf32>, tensor<32xf32>) : tensor<32xf32>, tensor<32xf32>
        flow.dispatch.tensor.store %4#0, %0, offsets = [0], sizes = [32], strides = [1] : tensor<32xf32> -> !flow.dispatch.tensor<readwrite:32xf32>
        flow.dispatch.tensor.store %4#1, %1, offsets = [0], sizes = [32], strides = [1] : tensor<32xf32> -> !flow.dispatch.tensor<readwrite:32xf32>
        return
      }
    }
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64]{{\]}}>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDefault>
//       CHECK: hal.executable.entry_point public @static_1d_fft_stage2
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: func @static_1d_fft_stage2()
//       CHECK:   iree_linalg_ext.fft
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @static_3d_fft_stage3  {
  hal.executable.variant @system_elf_x86_64, target = <"llvm", "system-elf-x86_64"> {
    hal.executable.entry_point @static_3d_fft_stage3 layout(#executable_layout)
    builtin.module {
      builtin.func @static_3d_fft_stage3() {
        %c3 = arith.constant 3 : index
        %cst = arith.constant dense<[1.000000e+00, 0.707106769, 6.12323426E-17, -0.707106769]> : tensor<4xf32>
        %cst_0 = arith.constant dense<[-0.000000e+00, -0.707106769, -1.000000e+00, -0.707106769]> : tensor<4xf32>
        %0 = bufferization.to_memref %cst_0 : memref<4xf32>
        %1 = bufferization.to_memref %cst : memref<4xf32>
        %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<64x128x32xf32>
        %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<64x128x32xf32>
        iree_linalg_ext.fft
            ins(%c3, %1, %0 : index, memref<4xf32>, memref<4xf32>)
            outs(%2, %3 : memref<64x128x32xf32>, memref<64x128x32xf32>)
        return
      }
    }
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 64]{{\]}}>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDefault>
//       CHECK: hal.executable.entry_point public @static_3d_fft_stage3
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: func @static_3d_fft_stage3()
//       CHECK:   iree_linalg_ext.fft
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @outs_fusion {
  hal.executable.variant @system_elf_x86_64, target = <"llvm", "system-elf-x86_64"> {
    hal.executable.entry_point @outs_fusion_fn layout(#executable_layout)
    builtin.module {
      builtin.func @outs_fusion_fn() {
        %cst = arith.constant 0.0 : f32
        %d0 = hal.interface.constant.load[0] : index
        %d1 = hal.interface.constant.load[1] : index
        %d2 = hal.interface.constant.load[2] : index
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:?x?xf32>{%d0, %d2}
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:?x?xf32>{%d2, %d1}
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:?x?xf32>{%d0, %d1}
        %init = linalg.init_tensor[%d0, %d1] : tensor<?x?xf32>
        %fill = linalg.generic {
              indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
              outs(%init : tensor<?x?xf32>) {
              ^bb0(%b0: f32):
                  linalg.yield %cst : f32
              } -> tensor<?x?xf32>
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [%d0, %d2], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:?x?xf32>{%d0, %d2} -> tensor<?x?xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [%d2, %d1], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:?x?xf32>{%d2, %d1} -> tensor<?x?xf32>
        %gemm = linalg.generic {
                indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                 affine_map<(d0, d1, d2) -> (d2, d1)>,
                                 affine_map<(d0, d1, d2) -> (d0, d1)>],
                iterator_types = ["parallel", "parallel", "reduction"]}
                ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
                outs(%fill : tensor<?x?xf32>) {
                ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
                  %6 = arith.mulf %arg0, %arg1 : f32
                  %7 = arith.addf %6, %arg2 : f32
                  linalg.yield %6 : f32
                } -> tensor<?x?xf32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
            : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>{%d0, %d1}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [1, 4, 0], [0, 0, 4]{{\]}}>
//      CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.entry_point public @outs_fusion_fn
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
//      CHECK: func @outs_fusion_fn()
//      CHECK:   linalg.generic
//  CHECK-NOT:   lowering_config
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @conv {
  hal.executable.variant public @system_elf_x86_64, target = <"llvm", "system-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }> {
    hal.executable.entry_point public @conv layout(#executable_layout)
    builtin.module {
      func @conv() {
        %N = hal.interface.constant.load[0] : index
        %H = hal.interface.constant.load[1] : index
        %W = hal.interface.constant.load[2] : index
        %C = hal.interface.constant.load[3] : index
        %R = hal.interface.constant.load[4] : index
        %S = hal.interface.constant.load[5] : index
        %F = hal.interface.constant.load[6] : index
        %P = hal.interface.constant.load[7] : index
        %Q = hal.interface.constant.load[8] : index
        %input_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:?x?x?x?xf32>{%N, %H, %W, %C}
        %filter_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:?x?x?x?xf32>{%R, %S, %C, %F}
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<readwrite:?x?x?x?xf32>{%N, %P, %Q, %F}
        %input = flow.dispatch.tensor.load %input_binding, offsets = [0, 0, 0, 0], sizes = [%N, %H, %W, %C], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:?x?x?x?xf32>{%N, %H, %W, %C} -> tensor<?x?x?x?xf32>
        %filter = flow.dispatch.tensor.load %filter_binding, offsets = [0, 0, 0, 0], sizes = [%R, %S, %C, %F], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:?x?x?x?xf32>{%R, %S, %C, %F} -> tensor<?x?x?x?xf32>
        %init = flow.dispatch.tensor.load %result_binding, offsets = [0, 0, 0, 0], sizes = [%N, %P, %Q, %F], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readwrite:?x?x?x?xf32>{%N, %P, %Q, %F} -> tensor<?x?x?x?xf32>
        %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
            ins(%input, %filter : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
            outs(%init : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
        flow.dispatch.tensor.store %conv, %result_binding, offsets = [0, 0, 0, 0], sizes = [%N, %P, %Q, %F], strides = [1, 1, 1, 1]
            : tensor<?x?x?x?xf32> -> !flow.dispatch.tensor<readwrite:?x?x?x?xf32>{%N, %P, %Q, %F}
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 64, 64, 64, 0, 0, 0], [1, 1, 8, 8, 0, 0, 0], [0, 0, 0, 0, 1, 1, 8]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
//      CHECK: hal.executable.entry_point public @conv
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:     linalg.conv_2d_nhwc_hwcf
//      CHECK:         lowering_config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @conv_static {
  hal.executable.variant public @system_elf_x86_64, target = <"llvm", "system-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }> {
    hal.executable.entry_point public @conv_static layout(#executable_layout)
    builtin.module {
      func @conv_static() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c607520 = arith.constant 607520 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:1x225x225x3xf32>
        %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c607520) alignment(32) : !flow.dispatch.tensor<readonly:3x3x3x16xf32>
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<writeonly:1x112x112x16xf32>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 225, 225, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:1x225x225x3xf32> -> tensor<1x225x225x3xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 3, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:3x3x3x16xf32> -> tensor<3x3x3x16xf32>
        %5 = linalg.init_tensor [1, 112, 112, 16] : tensor<1x112x112x16xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>
        %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x225x225x3xf32>, tensor<3x3x3x16xf32>) outs(%6 : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 112, 112, 16], strides = [1, 1, 1, 1] : tensor<1x112x112x16xf32> -> !flow.dispatch.tensor<writeonly:1x112x112x16xf32>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 28, 28, 16, 0, 0, 0], [1, 1, 4, 8, 0, 0, 0], [0, 0, 0, 0, 1, 1, 3]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
//      CHECK: hal.executable.entry_point public @conv_static
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:     linalg.conv_2d_nhwc_hwcf


// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @depthwise_conv_static {
  hal.executable.variant public @system_elf_x86_64, target = <"llvm", "system-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }> {
    hal.executable.entry_point public @depthwise_conv_static layout(#executable_layout)
    builtin.module {
      func @depthwise_conv_static() {
        %cst = arith.constant 0.0 : f32
        %input_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:1x161x161x96xf32>
        %filter_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:3x3x96xf32>
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:1x80x80x96xf32>
        %input = flow.dispatch.tensor.load %input_binding, offsets = [0, 0, 0, 0], sizes = [1, 161, 161, 96], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:1x161x161x96xf32> -> tensor<1x161x161x96xf32>
        %filter = flow.dispatch.tensor.load %filter_binding, offsets = [0, 0, 0], sizes = [3, 3, 96], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:3x3x96xf32> -> tensor<3x3x96xf32>
        %init = linalg.init_tensor [1, 80, 80, 96] : tensor<1x80x80x96xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x80x80x96xf32>) -> tensor<1x80x80x96xf32>
        %conv = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
            ins(%input, %filter : tensor<1x161x161x96xf32>, tensor<3x3x96xf32>) outs(%fill : tensor<1x80x80x96xf32>) -> tensor<1x80x80x96xf32>
        flow.dispatch.tensor.store %conv, %result_binding, offsets = [0, 0, 0, 0], sizes = [1, 80, 80, 96], strides = [1, 1, 1, 1]
            : tensor<1x80x80x96xf32> -> !flow.dispatch.tensor<writeonly:1x80x80x96xf32>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 20, 40, 48, 0, 0], [1, 1, 8, 8, 0, 0], [0, 0, 0, 0, 1, 3]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
//      CHECK: hal.executable.entry_point public @depthwise_conv_static
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:     linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:       lowering_config  = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @generic_static {
  hal.executable.variant public @system_elf_x86_64, target = <"llvm", "system-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 64 : index,
    target_triple = "x86_64-pc-linux-gnu"
  }> {
    hal.executable.entry_point public @generic_static layout(#executable_layout)
    builtin.module {
      func @generic_static() {
        %input_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:96x16xf32>
        %result_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:16x96xf32>
        %input = flow.dispatch.tensor.load %input_binding, offsets = [0, 0], sizes = [96, 16], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:96x16xf32> -> tensor<96x16xf32>
        %init = linalg.init_tensor [16, 96] : tensor<16x96xf32>
        %result = linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel"]}
            ins(%input : tensor<96x16xf32>) outs(%init : tensor<16x96xf32>) {
            ^bb0(%b0: f32, %b1: f32):  // no predecessors
              linalg.yield %b0 : f32
            } -> tensor<16x96xf32>
        flow.dispatch.tensor.store %result, %result_binding, offsets = [0, 0], sizes = [16, 96], strides = [1, 1]
            : tensor<16x96xf32> -> !flow.dispatch.tensor<writeonly:16x96xf32>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[16, 32], [16, 16], [0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.entry_point public @generic_static
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
//      CHECK:       lowering_config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_static {
  hal.executable.variant public @system_elf_arm_64, target = <"llvm", "system-elf-arm_64", {
    data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-linux-android30"
  }> {
    hal.executable.entry_point public @matmul_static layout(#executable_layout)
    builtin.module {
      func @matmul_static() {
        %cst = arith.constant 0.0 : f32
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:196x240xf32>
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:240x40xf32>
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:196x40xf32>
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [196, 240], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:196x240xf32> -> tensor<196x240xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [240, 40], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:240x40xf32> -> tensor<240x40xf32>
        %init = linalg.init_tensor [196, 40] : tensor<196x40xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<196x40xf32>) -> tensor<196x40xf32>
        %gemm = linalg.matmul ins(%lhs, %rhs : tensor<196x240xf32>, tensor<240x40xf32>)
            outs(%fill : tensor<196x40xf32>) -> tensor<196x40xf32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [196, 40], strides = [1, 1]
            : tensor<196x40xf32> -> !flow.dispatch.tensor<writeonly:196x40xf32>
        return
      }
    }
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[28, 8, 0], [4, 4, 60], [4, 4, 4]{{\]}}>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUTileFuseAndVectorize>
//       CHECK: hal.executable.entry_point public @matmul_static
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.matmul
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @restrict_num_workgroups {
  hal.executable.variant public @system_elf_arm_64, target = <"llvm", "system-elf-arm_64", {
    data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-linux-android30"
  }> {
    hal.executable.entry_point public @restrict_num_workgroups layout(#executable_layout)
    builtin.module {
      func @restrict_num_workgroups() {
        %cst = arith.constant 0.000000e+00 : f32
        %input_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:1x11x11x576xf32>
        %filter_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:5x5x576xf32>
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:1x7x7x576xf32>
        %input = flow.dispatch.tensor.load %input_binding, offsets = [0, 0, 0, 0], sizes = [1, 11, 11, 576], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:1x11x11x576xf32> -> tensor<1x11x11x576xf32>
        %filter = flow.dispatch.tensor.load %filter_binding, offsets = [0, 0, 0], sizes = [5, 5, 576], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:5x5x576xf32> -> tensor<5x5x576xf32>
        %init = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
        %conv = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
            ins(%input, %filter : tensor<1x11x11x576xf32>, tensor<5x5x576xf32>)
            outs(%fill : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
        flow.dispatch.tensor.store %conv, %result_binding, offsets = [0, 0, 0, 0], sizes = [1, 7, 7, 576], strides = [1, 1, 1, 1]
            : tensor<1x7x7x576xf32> -> !flow.dispatch.tensor<writeonly:1x7x7x576xf32>
        return
      }
    }
  }
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 1, 7, 64, 0, 0], [1, 1, 7, 8, 0, 0], [0, 0, 0, 0, 1, 1]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
//       CHECK: hal.executable.entry_point public @restrict_num_workgroups
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.depthwise_conv_2d_nhwc_hwc
//  CHECK-SAME:     lowering_config = #[[CONFIG]]


// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_x86  {
  hal.executable.variant public @embedded_elf_x86_64, target = #hal.executable.target<
    "llvm",
    "embedded-elf-x86_64", {
      data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
      native_vector_size = 16 : index,
      target_triple = "x86_64-unknown-unknown-eabi-elf"
    }> {
    hal.executable.entry_point public @matmul_x86 layout(#executable_layout)
    builtin.module {
      func @matmul_x86() {
        %cst = arith.constant 0.0 : f32
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:384x512xf32>
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:512x128xf32>
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:384x128xf32>
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [384, 512], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:384x512xf32> -> tensor<384x512xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [512, 128], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:512x128xf32> -> tensor<512x128xf32>
        %init = linalg.init_tensor [384, 128] : tensor<384x128xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<384x128xf32>) -> tensor<384x128xf32>
        %gemm = linalg.matmul ins(%lhs, %rhs : tensor<384x512xf32>, tensor<512x128xf32>)
            outs(%fill : tensor<384x128xf32>) -> tensor<384x128xf32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [384, 128], strides = [1, 1]
            : tensor<384x128xf32> -> !flow.dispatch.tensor<writeonly:384x128xf32>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] =  #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [8, 32, 0], [0, 0, 16]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.entry_point public @matmul_x86
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_riscv  {
  hal.executable.variant public @embedded_elf_x86_64, target = #hal.executable.target<
    "llvm",
    "embedded-elf-riscv_32", {
      cpu_features = "+m,+f",
      data_layout = "e-m:e-p:32:32-i64:64-n32-S128",
      native_vector_size = 0 : index,
      target_triple = "riscv32-unknown-unknown-eabi-elf"
    }> {
    hal.executable.entry_point public @matmul_riscv layout(#executable_layout)
    builtin.module {
      func @matmul_riscv() {
        %cst = arith.constant 0.0 : f32
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:384x512xf32>
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:512x128xf32>
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:384x128xf32>
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [384, 512], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:384x512xf32> -> tensor<384x512xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [512, 128], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:512x128xf32> -> tensor<512x128xf32>
        %init = linalg.init_tensor [384, 128] : tensor<384x128xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<384x128xf32>) -> tensor<384x128xf32>
        %gemm = linalg.matmul ins(%lhs, %rhs : tensor<384x512xf32>, tensor<512x128xf32>)
            outs(%fill : tensor<384x128xf32>) -> tensor<384x128xf32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [384, 128], strides = [1, 1]
            : tensor<384x128xf32> -> !flow.dispatch.tensor<writeonly:384x128xf32>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] =  #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [8, 32, 0], [0, 0, 16]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.entry_point public @matmul_riscv
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 4, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<
  "llvm", "embedded-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-unknown-eabi-elf"
  }
>
hal.executable private @reduction {
  hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.entry_point public @predict_dispatch_86 ordinal(0) layout(#executable_layout)
    builtin.module  {
      func @predict_dispatch_86(%arg0: !flow.dispatch.tensor<readonly:7x7x2048xf32>,
          %arg1: !flow.dispatch.tensor<writeonly:7xf32>) {
        %cst = arith.constant 0.0 : f32
        %cst1 = arith.constant 10.0 : f32
        %input = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [7, 7, 2048], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:7x7x2048xf32> -> tensor<7x7x2048xf32>
        %init = linalg.init_tensor [7] : tensor<7xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<7xf32>) -> tensor<7xf32> 
        %reduce = linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0)>],
            iterator_types = ["parallel", "reduction", "reduction"]}
            ins(%input : tensor<7x7x2048xf32>) outs(%fill : tensor<7xf32>) {
            ^bb0(%b0: f32, %b1: f32):
              %addf = arith.addf %b0, %b1 : f32
              linalg.yield %addf : f32
            } -> tensor<7xf32>
        %generic = linalg.generic {
            indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
            iterator_types = ["parallel"]}
            ins(%reduce : tensor<7xf32>) outs(%init : tensor<7xf32>) {
            ^bb0(%b0: f32, %b1: f32):
              %11 = arith.divf %b0, %cst1 : f32
              linalg.yield %11 : f32
            } -> tensor<7xf32>
          flow.dispatch.tensor.store %generic, %arg1, offsets = [0], sizes = [7], strides = [1]
              : tensor<7xf32> -> !flow.dispatch.tensor<writeonly:7xf32>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 0, 0], [1, 0, 0], [0, 1, 4]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.entry_point public @predict_dispatch_86
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic {indexing_maps = [#{{.+}}, #{{.+}}], iterator_types = ["parallel", "reduction", "reduction"]}
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_i8_i8_i32  {
  hal.executable.variant public @embedded_elf_x86_64, target = #hal.executable.target<
    "llvm",
    "embedded-elf-x86_64", {
      data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
      native_vector_size = 4 : index,
      target_triple = "x86_64-unknown-unknown-eabi-elf"
    }> {
    hal.executable.entry_point public @matmul_i8_i8_i32 layout(#executable_layout)
    builtin.module {
      func @matmul_i8_i8_i32() {
        %c0 = arith.constant 0 : index
        %M = hal.interface.constant.load[0] : index
        %N = hal.interface.constant.load[1] : index
        %K = hal.interface.constant.load[2] : index
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32)
            : !flow.dispatch.tensor<readonly:?x?xi8>{%M, %K}
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32)
            : !flow.dispatch.tensor<readonly:?x?xi8>{%K, %N}
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(32)
            : !flow.dispatch.tensor<readwrite:?x?xi32>{%M, %N}
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:?x?xi8>{%M, %K} -> tensor<?x?xi8>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:?x?xi8>{%K, %N} -> tensor<?x?xi8>
        %init = flow.dispatch.tensor.load %result_binding, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
            : !flow.dispatch.tensor<readwrite:?x?xi32>{%M, %N} -> tensor<?x?xi32>
        %gemm = linalg.matmul ins(%lhs, %rhs : tensor<?x?xi8>, tensor<?x?xi8>) outs(%init : tensor<?x?xi32>) -> tensor<?x?xi32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
            : tensor<?x?xi32> -> !flow.dispatch.tensor<readwrite:?x?xi32>{%M, %N}
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [8, 32, 0], [0, 0, 16]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.entry_point public @matmul_i8_i8_i32
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 4, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<
  "llvm", "embedded-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-unknown-eabi-elf"
  }
>
#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0, s1] -> (s1, -d0 + s0)>
hal.executable private @gemm_unit_N {
  hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.entry_point public @gemm_unit_N ordinal(0) layout(#executable_layout)
    builtin.module  {
      func @gemm_unit_N() {
        %c0 = arith.constant 0 : index
        %M = hal.interface.constant.load[0] : index
        %K = hal.interface.constant.load[1] : index
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32)
            : !flow.dispatch.tensor<readonly:?x?xf32>{%M, %K}
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32)
            : !flow.dispatch.tensor<readonly:?x1xf32>{%K}
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(32)
            : !flow.dispatch.tensor<readwrite:?x1xf32>{%M}
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [%K, 1], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:?x1xf32>{%K} -> tensor<?x1xf32>
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:?x?xf32>{%M, %K} -> tensor<?x?xf32>
        %init = flow.dispatch.tensor.load %result_binding, offsets = [0, 0], sizes = [%M, 1], strides = [1, 1]
            : !flow.dispatch.tensor<readwrite:?x1xf32>{%M} -> tensor<?x1xf32>
        %gemm = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x1xf32>) outs(%init : tensor<?x1xf32>) -> tensor<?x1xf32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [%M, 1], strides = [1, 1]
            : tensor<?x1xf32> -> !flow.dispatch.tensor<readwrite:?x1xf32>{%M}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 0, 0], [8, 0, 0], [0, 0, 16]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.entry_point public @gemm_unit_N
// CHECK-SAME:       translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 4, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<
  "llvm", "embedded-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-unknown-eabi-elf"
  }
>
hal.executable private @gemm_unit_M_unit_N {
  hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.entry_point public @gemm_unit_M_unit_N ordinal(0) layout(#executable_layout)
    builtin.module  {
      func @gemm_unit_M_unit_N() {
        %c0 = arith.constant 0 : index
        %K = hal.interface.constant.load[0] : index
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:1x?xf32>{%K}
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:?x1xf32>{%K}
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readwrite:1x1xf32>
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [1, %K], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:1x?xf32>{%K} -> tensor<1x?xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [%K, 1], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:?x1xf32>{%K} -> tensor<?x1xf32>
        %init = flow.dispatch.tensor.load %result_binding, offsets = [0, 0], sizes = [1, 1], strides = [1, 1]
            : !flow.dispatch.tensor<readwrite:1x1xf32> -> tensor<1x1xf32>
        %gemm = linalg.matmul ins(%lhs, %rhs : tensor<1x?xf32>, tensor<?x1xf32>) outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [1, 1], strides = [1, 1]
            : tensor<1x1xf32> -> !flow.dispatch.tensor<readwrite:1x1xf32>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 0, 0], [0, 0, 0], [0, 0, 16]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.entry_point public @gemm_unit_M_unit_N
// CHECK-SAME:       translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 4, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<
  "llvm", "embedded-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-unknown-eabi-elf"
  }
>
hal.executable private @matmul_odd {
  hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.entry_point public @matmul_odd ordinal(0) layout(#executable_layout)
    builtin.module {
      func @matmul_odd() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:33x16xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:16x49xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:33x49xf32>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<writeonly:33x49xf32>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [33, 16], strides = [1, 1] : !flow.dispatch.tensor<readonly:33x16xf32> -> tensor<33x16xf32>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 49], strides = [1, 1] : !flow.dispatch.tensor<readonly:16x49xf32> -> tensor<16x49xf32>
        %6 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [33, 49], strides = [1, 1] : !flow.dispatch.tensor<readonly:33x49xf32> -> tensor<33x49xf32>
        %7 = linalg.init_tensor [33, 49] : tensor<33x49xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<33x49xf32>) -> tensor<33x49xf32>
        %9 = linalg.matmul ins(%4, %5 : tensor<33x16xf32>, tensor<16x49xf32>) outs(%8 : tensor<33x49xf32>) -> tensor<33x49xf32>
        flow.dispatch.tensor.store %9, %3, offsets = [0, 0], sizes = [33, 49], strides = [1, 1] : tensor<33x49xf32> -> !flow.dispatch.tensor<writeonly:33x49xf32>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[3, 7, 0], [3, 7, 0], [0, 0, 16]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.entry_point public @matmul_odd
// CHECK-SAME:       translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @generic_unit_dims {
  hal.executable.variant @llvm, target = <"llvm", "embedded-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }> {
    hal.executable.entry_point @generic_unit_dims layout(#executable_layout)
    builtin.module {
      func @generic_unit_dims() {
        %c0 = arith.constant 0 : index
        %d0 = hal.interface.constant.load[0] : index
        %d1 = hal.interface.constant.load[1] : index
        %d2 = hal.interface.constant.load[2] : index
        %d3 = hal.interface.constant.load[3] : index
        %in_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:1x?x1x1x?x?x1x?xf32>{%d0, %d1, %d2, %d3}
        %result_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:1x?x1x1x?x?x1x?xf32>{%d0, %d1, %d2, %d3}
        %in = flow.dispatch.tensor.load %in_binding, offsets=[0, 0, 0, 0, 0, 0, 0, 0],
            sizes=[1, %d0, 1, 1, %d1, %d2, 1, %d3], strides=[1, 1, 1, 1, 1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:1x?x1x1x?x?x1x?xf32>{%d0, %d1, %d2, %d3} -> tensor<1x?x1x1x?x?x1x?xf32>
        %init = linalg.init_tensor [1, %d0, 1, 1, %d1, %d2, 1, %d3] : tensor<1x?x1x1x?x?x1x?xf32>
        %generic = linalg.generic {
          indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>,
                           affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>],
          iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}
          ins(%in : tensor<1x?x1x1x?x?x1x?xf32>) outs(%init : tensor<1x?x1x1x?x?x1x?xf32>) {
          ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
            %7 = arith.addf %arg0, %arg0 : f32
            linalg.yield %7 : f32
          } -> tensor<1x?x1x1x?x?x1x?xf32>
        flow.dispatch.tensor.store %generic, %result_binding, offsets = [0, 0, 0, 0, 0, 0, 0, 0],
            sizes = [1, %d0, 1, 1, %d1, %d2, 1, %d3], strides = [1, 1, 1, 1, 1, 1, 1, 1]
            : tensor<1x?x1x1x?x?x1x?xf32> -> !flow.dispatch.tensor<writeonly:1x?x1x1x?x?x1x?xf32>{%d0, %d1, %d2, %d3}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 0, 0, 0, 64, 64, 0, 64], [0, 1, 0, 0, 1, 1, 0, 4], [0, 0, 0, 0, 0, 0, 0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.entry_point public @generic_unit_dim
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @reduce_to_scalar {
  hal.executable.variant @llvm, target = <"llvm", "embedded-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }> {
    hal.executable.entry_point @reduce_to_scalar layout(#executable_layout)
    builtin.module {
      func @reduce_to_scalar() {
        %c0 = arith.constant 0 : index
        %d0 = hal.interface.constant.load[0] : index
        %in_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:?xf32>{%d0}
        %out_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readwrite:f32>
        %in = flow.dispatch.tensor.load %in_binding, offsets=[0], sizes=[%d0], strides=[1] : !flow.dispatch.tensor<readonly:?xf32>{%d0} -> tensor<?xf32>
        %out = flow.dispatch.tensor.load %out_binding, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readwrite:f32> -> tensor<f32>
        %reduce = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
          iterator_types = ["reduction"]}
          ins(%in : tensor<?xf32>) outs(%out : tensor<f32>) {
          ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
            %7 = arith.addf %arg0, %arg1 : f32
            linalg.yield %7 : f32
          } -> tensor<f32>
        flow.dispatch.tensor.store %reduce, %out_binding, offsets = [], sizes = [], strides = [] : tensor<f32> -> !flow.dispatch.tensor<readwrite:f32>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0], [0], [4]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.entry_point public @reduce_to_scalar
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @scalar {
  hal.executable.variant @llvm, target = <"llvm", "embedded-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }> {
    hal.executable.entry_point @scalar layout(#executable_layout)
    builtin.module {
      func @scalar() {
        %c0 = arith.constant 0 : index
        %in_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:f32>
        %out_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:f32>
        %in = flow.dispatch.tensor.load %in_binding, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:f32> -> tensor<f32>
        %out = flow.dispatch.tensor.load %out_binding, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<writeonly:f32> -> tensor<f32>
        %reduce = linalg.generic {
          indexing_maps = [affine_map<() -> ()>,
                           affine_map<() -> ()>],
          iterator_types = []}
          ins(%in : tensor<f32>) outs(%out : tensor<f32>) {
          ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
            %7 = arith.addf %arg0, %arg1 : f32
            linalg.yield %7 : f32
          } -> tensor<f32>
        flow.dispatch.tensor.store %reduce, %out_binding, offsets = [], sizes = [], strides = [] : tensor<f32> -> !flow.dispatch.tensor<writeonly:f32>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDefault>
//      CHECK: hal.executable.entry_point public @scalar
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
