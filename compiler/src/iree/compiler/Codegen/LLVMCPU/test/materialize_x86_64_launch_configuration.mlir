// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target{test-lowering-configuration=true})))' --split-input-file %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matvec_static  {
  hal.executable.variant @llvm, target = <"llvm-cpu", "embedded-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }> {
    hal.executable.export @matvec_static layout(#pipeline_layout)
    builtin.module {
      func.func @matvec_static() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<128x384xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<384xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 384], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x384xf32>> -> tensor<128x384xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [384], strides = [1] : !flow.dispatch.tensor<readonly:tensor<384xf32>> -> tensor<384xf32>
        %5 = tensor.empty() : tensor<128xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128xf32>) -> tensor<128xf32>
        %7 = linalg.matvec ins(%3, %4 : tensor<128x384xf32>, tensor<384xf32>) outs(%6 : tensor<128xf32>) -> tensor<128xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0], sizes = [128], strides = [1] : tensor<128xf32> -> !flow.dispatch.tensor<writeonly:tensor<128xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 0], [32, 0], [0, 16]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPadExpert>
//      CHECK: hal.executable.export public @matvec_static
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matvec
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matvec_dynamic  {
  hal.executable.variant @llvm, target = <"llvm-cpu", "embedded-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }> {
    hal.executable.export @matvec_dynamic layout(#pipeline_layout)
    builtin.module {
      func.func @matvec_dynamic() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = arith.index_cast %0 : i32 to index
        %4 = arith.index_cast %1 : i32 to index
        %5 = arith.index_cast %2 : i32 to index
        %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%3, %4}
        %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%5}
        %8 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?xf32>>{%3}
        %9 = hal.interface.constant.load[0] : i32
        %10 = arith.index_cast %9 : i32 to index
        %11 = flow.dispatch.tensor.load %8, offsets = [0], sizes = [%10], strides = [1] : !flow.dispatch.tensor<writeonly:tensor<?xf32>>{%3} -> tensor<?xf32>
        %12 = flow.dispatch.tensor.load %6, offsets = [0, 0], sizes = [%3, %4], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%3, %4} -> tensor<?x?xf32>
        %13 = flow.dispatch.tensor.load %7, offsets = [0], sizes = [%5], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%5} -> tensor<?xf32>
        %14 = linalg.fill ins(%cst : f32) outs(%11 : tensor<?xf32>) -> tensor<?xf32>
        %15 = linalg.matvec ins(%12, %13 : tensor<?x?xf32>, tensor<?xf32>) outs(%14 : tensor<?xf32>) -> tensor<?xf32>
        flow.dispatch.tensor.store %15, %8, offsets = [0], sizes = [%3], strides = [1] : tensor<?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?xf32>>{%3}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[128, 0], [32, 0], [0, 1]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPeelingExpert>
//      CHECK: hal.executable.export public @matvec_dynamic
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matvec
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @dot_static  {
  hal.executable.variant @llvm, target = <"llvm-cpu", "embedded-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }> {
    hal.executable.export @dot_static layout(#pipeline_layout)
    builtin.module {
      func.func @dot_static() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<384xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<384xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<f32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [384], strides = [1] : !flow.dispatch.tensor<readonly:tensor<384xf32>> -> tensor<384xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [384], strides = [1] : !flow.dispatch.tensor<readonly:tensor<384xf32>> -> tensor<384xf32>
        %5 = tensor.empty() : tensor<f32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<f32>) -> tensor<f32>
        %7 = linalg.dot ins(%3, %4 : tensor<384xf32>, tensor<384xf32>) outs(%6 : tensor<f32>) -> tensor<f32>
        flow.dispatch.tensor.store %7, %2, offsets = [], sizes = [], strides = [] : tensor<f32> -> !flow.dispatch.tensor<writeonly:tensor<f32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0], [0], [16]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPadExpert>
//      CHECK: hal.executable.export public @dot_static
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.dot
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @dot_dynamic  {
  hal.executable.variant @llvm, target = <"llvm-cpu", "embedded-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }> {
    hal.executable.export @dot_dynamic layout(#pipeline_layout)
    builtin.module {
      func.func @dot_dynamic() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = arith.index_cast %0 : i32 to index
        %3 = arith.index_cast %1 : i32 to index
        %4 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<f32>>
        %5 = flow.dispatch.tensor.load %4, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<writeonly:tensor<f32>> -> tensor<f32>
        %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%2}
        %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%3}
        %8 = flow.dispatch.tensor.load %6, offsets = [0], sizes = [%2], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%2} -> tensor<?xf32>
        %9 = flow.dispatch.tensor.load %7, offsets = [0], sizes = [%3], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%3} -> tensor<?xf32>
        %10 = linalg.fill ins(%cst : f32) outs(%5 : tensor<f32>) -> tensor<f32>
        %11 = linalg.dot ins(%8, %9 : tensor<?xf32>, tensor<?xf32>) outs(%10 : tensor<f32>) -> tensor<f32>
        flow.dispatch.tensor.store %11, %4, offsets = [], sizes = [], strides = [] : tensor<f32> -> !flow.dispatch.tensor<writeonly:tensor<f32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0], [0], [16]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPeelingExpert>
//      CHECK: hal.executable.export public @dot_dynamic
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.dot
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @add {
  hal.executable.variant @llvm, target = <"llvm-cpu", "embedded-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }> {
    hal.executable.export @add layout(#pipeline_layout)
    builtin.module {
      func.func @add() {
        %c0 = arith.constant 0 : index
        %dim0 = hal.interface.constant.load[0] : index
        %dim1 = hal.interface.constant.load[1] : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%dim0, %dim1}
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%dim1}
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%dim0, %dim1}
        %3 = flow.dispatch.tensor.load %0, offsets=[0, 0], sizes=[%dim0, %dim1], strides=[1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%dim0, %dim1} -> tensor<?x?xf32>
        %4 = flow.dispatch.tensor.load %1, offsets=[0], sizes=[%dim1], strides=[1] : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%dim1} -> tensor<?xf32>
        %5 = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
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
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%dim0, %dim1], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%dim0, %dim1}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64], [1, 4], [0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPeelingExpert>
//      CHECK: hal.executable.export public @add
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
hal.executable private @add4D  {
  hal.executable.variant @llvm, target = <"llvm-cpu", "embedded-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }> {
    hal.executable.export @add4D layout(#pipeline_layout)
    builtin.module {
      func.func @add4D() {
        %d0 = hal.interface.constant.load[0] : index
        %d1 = hal.interface.constant.load[1] : index
        %d2 = hal.interface.constant.load[2] : index
        %d3 = hal.interface.constant.load[3] : index
        %arg1_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32)
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%d0, %d1, %d2, %d3}
        %arg2_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32)
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%d0, %d1, %d2, %d3}
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32)
            : !flow.dispatch.tensor<writeonly:tensor<?x?x?x?xf32>>{%d0, %d1, %d2, %d3}
        %arg1 = flow.dispatch.tensor.load %arg1_binding, offsets = [0, 0, 0, 0], sizes = [%d0, %d1, %d2, %d3], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%d0, %d1, %d2, %d3} -> tensor<?x?x?x?xf32>
        %arg2 = flow.dispatch.tensor.load %arg2_binding, offsets = [0, 0, 0, 0], sizes = [%d0, %d1, %d2, %d3], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%d0, %d1, %d2, %d3} -> tensor<?x?x?x?xf32>
        %init = tensor.empty(%d0, %d1, %d2, %d3) : tensor<?x?x?x?xf32>
        %add = linalg.generic {
            indexing_maps = [#map, #map, #map],
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
            ins(%arg1, %arg2 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%init : tensor<?x?x?x?xf32>) {
            ^bb0(%b0: f32, %b1: f32, %b2: f32):  // no predecessors
              %addf = arith.addf %b0, %b1 : f32
              linalg.yield %addf : f32
            } -> tensor<?x?x?x?xf32>
        flow.dispatch.tensor.store %add, %result_binding, offsets = [0, 0, 0, 0], sizes = [%d0, %d1, %d2, %d3], strides = [1, 1, 1, 1]
            : tensor<?x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x?x?xf32>>{%d0, %d1, %d2, %d3}
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 64, 64, 64], [1, 1, 1, 4], [0, 0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPeelingExpert>
//      CHECK: hal.executable.export public @add4D
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @add_static {
  hal.executable.variant @llvm, target = <"llvm-cpu", "embedded-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }> {
    hal.executable.export @add_static layout(#pipeline_layout)
    builtin.module {
      func.func @add_static() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<64x16x32x128xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x16x32x128xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [64, 16, 32, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x16x32x128xf32>> -> tensor<64x16x32x128xf32>
        %3 = tensor.empty() : tensor<64x16x32x128xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<64x16x32x128xf32>) outs(%3 : tensor<64x16x32x128xf32>) {
        ^bb0(%arg0: f32, %arg1: f32):
          %5 = arith.addf %arg0, %arg0 : f32
          linalg.yield %5 : f32
        } -> tensor<64x16x32x128xf32>
        flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [64, 16, 32, 128], strides = [1, 1, 1, 1] : tensor<64x16x32x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x16x32x128xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 8, 16, 64], [1, 1, 1, 4], [0, 0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPeelingExpert>
//      CHECK: hal.executable.export public @add_static
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#compilation = #iree_codegen.compilation_info<
    lowering_config = <tile_sizes = [[64, 64, 0], [32, 32, 0], [0, 0, 32]]>,
    translation_info  = <CPUDoubleTilingPadExpert>,
    workgroup_size = []>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @preset_config_matmul_tensors  {
  hal.executable.variant @system_elf_x86_64, target = <"llvm-cpu", "system-elf-x86_64"> {
    hal.executable.export @preset_config layout(#pipeline_layout)
    builtin.module {
      func.func @preset_config() {
        %cst = arith.constant 0.000000e+00 : f32
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<128x256xf32>>
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<256x512xf32>>
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [128, 256], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<128x256xf32>> -> tensor<128x256xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [256, 512], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<256x512xf32>> -> tensor<256x512xf32>
        %init = tensor.empty() : tensor<128x512xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<128x512xf32>) -> tensor<128x512xf32>
        %gemm = linalg.matmul {compilation_info = #compilation}
            ins(%lhs, %rhs : tensor<128x256xf32>, tensor<256x512xf32>)
            outs(%fill : tensor<128x512xf32>) -> tensor<128x512xf32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [128, 512], strides = [1, 1]
            : tensor<128x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [32, 32, 0], [0, 0, 32]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPadExpert>
//      CHECK: hal.executable.export
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: func.func @preset_config
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @copy_op_dynamic {
  hal.executable.variant @system_elf_x86_64, target = <"llvm-cpu", "system-elf-x86_64"> {
    hal.executable.export @copy_op_dynamic layout(#pipeline_layout)
    builtin.module {
      func.func @copy_op_dynamic() {
        %d0 = hal.interface.constant.load[0] : index
        %d1 = hal.interface.constant.load[1] : index
        %d2 = hal.interface.constant.load[2] : index
        %d3 = hal.interface.constant.load[3] : index
        %o0 = hal.interface.constant.load[4] : index
        %o1 = hal.interface.constant.load[5] : index
        %source = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<?x?xi32>{%d0, %d1}
        %dest = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<?x?xi32>{%d2, %d3}
        %dest_view = memref.subview %dest[%o0, %o1] [%d0, %d1] [1, 1] : memref<?x?xi32> to memref<?x?xi32,  strided<[?, ?], offset : ?>>
        linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d0, d1)> , affine_map<(d0, d1) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel"]}
            ins(%source : memref<?x?xi32>) outs(%dest_view : memref<?x?xi32, strided<[?, ?], offset : ?>>) {
          ^bb0(%arg0 : i32, %arg1 : i32):
            linalg.yield %arg0 : i32
          }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64], [1, 4], [0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUBufferOpsTileAndVectorize>
//      CHECK: hal.executable.export public @copy_op_dynamic
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @static_1d_fft_stage2  {
  hal.executable.variant @system_elf_x86_64, target = <"llvm-cpu", "system-elf-x86_64"> {
    hal.executable.export @static_1d_fft_stage2 layout(#pipeline_layout)
    builtin.module {
      func.func @static_1d_fft_stage2() {
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %cst = arith.constant dense<[1.000000e+00, 6.12323426E-17]> : tensor<2xf32>
        %cst_0 = arith.constant dense<[-0.000000e+00, -1.000000e+00]> : tensor<2xf32>
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<32xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<32xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [32], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<32xf32>> -> tensor<32xf32>
        %3 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [32], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<32xf32>> -> tensor<32xf32>
        %4:2 = iree_linalg_ext.fft {__internal_linalg_transform__ = "workgroup"} ins(%c2, %cst, %cst_0 : index, tensor<2xf32>, tensor<2xf32>) outs(%2, %3 : tensor<32xf32>, tensor<32xf32>) : tensor<32xf32>, tensor<32xf32>
        flow.dispatch.tensor.store %4#0, %0, offsets = [0], sizes = [32], strides = [1] : tensor<32xf32> -> !flow.dispatch.tensor<readwrite:tensor<32xf32>>
        flow.dispatch.tensor.store %4#1, %1, offsets = [0], sizes = [32], strides = [1] : tensor<32xf32> -> !flow.dispatch.tensor<readwrite:tensor<32xf32>>
        return
      }
    }
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64]{{\]}}>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDefault>
//       CHECK: hal.executable.export public @static_1d_fft_stage2
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: func.func @static_1d_fft_stage2()
//       CHECK:   iree_linalg_ext.fft
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @static_3d_fft_stage3  {
  hal.executable.variant @system_elf_x86_64, target = <"llvm-cpu", "system-elf-x86_64"> {
    hal.executable.export @static_3d_fft_stage3 layout(#pipeline_layout)
    builtin.module {
      func.func @static_3d_fft_stage3() {
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
//       CHECK: hal.executable.export public @static_3d_fft_stage3
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: func.func @static_3d_fft_stage3()
//       CHECK:   iree_linalg_ext.fft
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @outs_fusion {
  hal.executable.variant @system_elf_x86_64, target = <"llvm-cpu", "system-elf-x86_64"> {
    hal.executable.export @outs_fusion_fn layout(#pipeline_layout)
    builtin.module {
      func.func @outs_fusion_fn() {
        %cst = arith.constant 0.0 : f32
        %d0 = hal.interface.constant.load[0] : index
        %d1 = hal.interface.constant.load[1] : index
        %d2 = hal.interface.constant.load[2] : index
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%d0, %d2}
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%d2, %d1}
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%d0, %d1}
        %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
        %fill = linalg.generic {
              indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
              outs(%init : tensor<?x?xf32>) {
              ^bb0(%b0: f32):
                  linalg.yield %cst : f32
              } -> tensor<?x?xf32>
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [%d0, %d2], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%d0, %d2} -> tensor<?x?xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [%d2, %d1], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%d2, %d1} -> tensor<?x?xf32>
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
            : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%d0, %d1}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32, 32, 0], [1, 4, 0], [0, 0, 1]{{\]}}>
//      CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPeelingExpert>
//      CHECK: hal.executable.export public @outs_fusion_fn
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
//      CHECK: func.func @outs_fusion_fn()
//      CHECK:   linalg.generic
//  CHECK-NOT:   lowering_config
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @conv_dynamic {
  hal.executable.variant public @system_elf_x86_64, target = <"llvm-cpu", "system-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }> {
    hal.executable.export public @conv_dynamic layout(#pipeline_layout)
    builtin.module {
      func.func @conv_dynamic() {
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
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%N, %H, %W, %C}
        %filter_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%R, %S, %C, %F}
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<readwrite:tensor<?x?x?x?xf32>>{%N, %P, %Q, %F}
        %input = flow.dispatch.tensor.load %input_binding, offsets = [0, 0, 0, 0], sizes = [%N, %H, %W, %C], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%N, %H, %W, %C} -> tensor<?x?x?x?xf32>
        %filter = flow.dispatch.tensor.load %filter_binding, offsets = [0, 0, 0, 0], sizes = [%R, %S, %C, %F], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%R, %S, %C, %F} -> tensor<?x?x?x?xf32>
        %init = flow.dispatch.tensor.load %result_binding, offsets = [0, 0, 0, 0], sizes = [%N, %P, %Q, %F], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readwrite:tensor<?x?x?x?xf32>>{%N, %P, %Q, %F} -> tensor<?x?x?x?xf32>
        %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
            ins(%input, %filter : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
            outs(%init : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
        flow.dispatch.tensor.store %conv, %result_binding, offsets = [0, 0, 0, 0], sizes = [%N, %P, %Q, %F], strides = [1, 1, 1, 1]
            : tensor<?x?x?x?xf32> -> !flow.dispatch.tensor<readwrite:tensor<?x?x?x?xf32>>{%N, %P, %Q, %F}
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 64, 64, 64, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
//      CHECK: hal.executable.export public @conv_dynamic
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:     linalg.conv_2d_nhwc_hwcf
//      CHECK:         lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @conv_static {
  hal.executable.variant public @system_elf_x86_64, target = <"llvm-cpu", "system-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }> {
    hal.executable.export public @conv_static layout(#pipeline_layout)
    builtin.module {
      func.func @conv_static() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c607520 = arith.constant 607520 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x225x225x3xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) offset(%c607520) : !flow.dispatch.tensor<readonly:tensor<3x3x3x16xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 225, 225, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x225x225x3xf32>> -> tensor<1x225x225x3xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 3, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x3x16xf32>> -> tensor<3x3x3x16xf32>
        %5 = tensor.empty() : tensor<1x112x112x16xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>
        %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x225x225x3xf32>, tensor<3x3x3x16xf32>) outs(%6 : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 112, 112, 16], strides = [1, 1, 1, 1] : tensor<1x112x112x16xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 28, 28, 16, 0, 0, 0], [1, 1, 4, 8, 0, 0, 0], [0, 0, 0, 0, 1, 1, 3]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
//      CHECK: hal.executable.export public @conv_static
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:     linalg.conv_2d_nhwc_hwcf


// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-unknown-eabi-elf"}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @conv_nchw_static {
  hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.export public @conv_nchw_static ordinal(0) layout(#pipeline_layout)
    builtin.module {
      func.func @conv_nchw_static() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x128x30x30xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<128x128x3x3xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x128x28x28xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 128, 30, 30], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x128x30x30xf32>> -> tensor<1x128x30x30xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [128, 128, 3, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x128x3x3xf32>> -> tensor<128x128x3x3xf32>
        %5 = tensor.empty() : tensor<1x128x28x28xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
        %7 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%3, %4 : tensor<1x128x30x30xf32>, tensor<128x128x3x3xf32>) outs(%6 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 128, 28, 28], strides = [1, 1, 1, 1] : tensor<1x128x28x28xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x128x28x28xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 64, 28, 4, 0, 0, 0], [1, 8, 1, 4, 0, 0, 0], [0, 0, 0, 0, 8, 1, 1]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
//      CHECK: hal.executable.export public @conv_nchw_static
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:     linalg.conv_2d_nchw_fchw

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @depthwise_conv_static {
  hal.executable.variant public @system_elf_x86_64, target = <"llvm-cpu", "system-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 64 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }> {
    hal.executable.export public @depthwise_conv_static layout(#pipeline_layout)
    builtin.module {
      func.func @depthwise_conv_static() {
        %cst = arith.constant 0.0 : f32
        %input_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<1x161x161x240xf32>>
        %filter_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<3x3x240xf32>>
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<1x80x80x240xf32>>
        %input = flow.dispatch.tensor.load %input_binding, offsets = [0, 0, 0, 0], sizes = [1, 161, 161, 240], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1x161x161x240xf32>> -> tensor<1x161x161x240xf32>
        %filter = flow.dispatch.tensor.load %filter_binding, offsets = [0, 0, 0], sizes = [3, 3, 240], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<3x3x240xf32>> -> tensor<3x3x240xf32>
        %init = tensor.empty() : tensor<1x80x80x240xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x80x80x240xf32>) -> tensor<1x80x80x240xf32>
        %conv = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
            ins(%input, %filter : tensor<1x161x161x240xf32>, tensor<3x3x240xf32>) outs(%fill : tensor<1x80x80x240xf32>) -> tensor<1x80x80x240xf32>
        flow.dispatch.tensor.store %conv, %result_binding, offsets = [0, 0, 0, 0], sizes = [1, 80, 80, 240], strides = [1, 1, 1, 1]
            : tensor<1x80x80x240xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x80x80x240xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 40, 40, 48, 0, 0], [1, 1, 8, 16, 0, 0], [0, 0, 0, 0, 1, 3]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
//      CHECK: hal.executable.export public @depthwise_conv_static
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:     linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:       lowering_config  = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @thin_depthwise_conv_static {
  hal.executable.variant public @system_elf_x86_64, target = <"llvm-cpu", "system-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 64 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }> {
    hal.executable.export public @thin_depthwise_conv_static layout(#pipeline_layout)
    builtin.module {
      func.func @thin_depthwise_conv_static() {
        %cst = arith.constant 0.0 : f32
        %input_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<1x57x57x72xf32>>
        %filter_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<3x3x72xf32>>
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<1x28x28x72xf32>>
        %input = flow.dispatch.tensor.load %input_binding, offsets = [0, 0, 0, 0], sizes = [1, 161, 161, 240], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1x57x57x72xf32>> -> tensor<1x57x57x72xf32>
        %filter = flow.dispatch.tensor.load %filter_binding, offsets = [0, 0, 0], sizes = [3, 3, 240], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<3x3x72xf32>> -> tensor<3x3x72xf32>
        %init = tensor.empty() : tensor<1x28x28x72xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x28x28x72xf32>) -> tensor<1x28x28x72xf32>
        %conv = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
          ins(%input, %filter : tensor<1x57x57x72xf32>, tensor<3x3x72xf32>)
          outs(%fill : tensor<1x28x28x72xf32>) -> tensor<1x28x28x72xf32>

        flow.dispatch.tensor.store %conv, %result_binding, offsets = [0, 0, 0, 0], sizes = [1, 28, 28, 72], strides = [1, 1, 1, 1]
            : tensor<1x28x28x72xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x28x28x72xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 7, 14, 36, 0, 0], [1, 1, 7, 18, 0, 0], [0, 0, 0, 0, 1, 3]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
//      CHECK: hal.executable.export public @thin_depthwise_conv_static
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:     linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:       lowering_config  = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @generic_static {
  hal.executable.variant public @system_elf_x86_64, target = <"llvm-cpu", "system-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 64 : index,
    target_triple = "x86_64-pc-linux-gnu"
  }> {
    hal.executable.export public @generic_static layout(#pipeline_layout)
    builtin.module {
      func.func @generic_static() {
        %input_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<96x16xf32>>
        %result_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<16x96xf32>>
        %input = flow.dispatch.tensor.load %input_binding, offsets = [0, 0], sizes = [96, 16], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<96x16xf32>> -> tensor<96x16xf32>
        %init = tensor.empty() : tensor<16x96xf32>
        %result = linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel"]}
            ins(%input : tensor<96x16xf32>) outs(%init : tensor<16x96xf32>) {
            ^bb0(%b0: f32, %b1: f32):  // no predecessors
              linalg.yield %b0 : f32
            } -> tensor<16x96xf32>
        flow.dispatch.tensor.store %result, %result_binding, offsets = [0, 0], sizes = [16, 96], strides = [1, 1]
            : tensor<16x96xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x96xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[16, 96], [16, 16], [0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPeelingExpert>
//      CHECK: hal.executable.export public @generic_static
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
//      CHECK:       lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_static  {
  hal.executable.variant public @embedded_elf_x86_64, target = #hal.executable.target<
    "llvm-cpu",
    "embedded-elf-x86_64", {
      data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
      native_vector_size = 16 : index,
      target_triple = "x86_64-unknown-unknown-eabi-elf"
    }> {
    hal.executable.export public @matmul_static layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_static() {
        %cst = arith.constant 0.0 : f32
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<384x512xf32>>
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<512x128xf32>>
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<384x128xf32>>
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [384, 512], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<384x512xf32>> -> tensor<384x512xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [512, 128], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<512x128xf32>> -> tensor<512x128xf32>
        %init = tensor.empty() : tensor<384x128xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<384x128xf32>) -> tensor<384x128xf32>
        %gemm = linalg.matmul ins(%lhs, %rhs : tensor<384x512xf32>, tensor<512x128xf32>)
            outs(%fill : tensor<384x128xf32>) -> tensor<384x128xf32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [384, 128], strides = [1, 1]
            : tensor<384x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<384x128xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] =  #iree_codegen.lowering_config<tile_sizes = {{\[}}[128, 64, 0], [8, 32, 0], [0, 0, 16]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPadExpert>
//      CHECK: hal.executable.export public @matmul_static
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 4, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<
  "llvm-cpu", "embedded-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-unknown-eabi-elf"
  }
>
hal.executable private @reduction {
  hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.export public @predict_dispatch_86 ordinal(0) layout(#pipeline_layout)
    builtin.module  {
      func.func @predict_dispatch_86(%arg0: !flow.dispatch.tensor<readonly:tensor<7x7x2048xf32>>,
          %arg1: !flow.dispatch.tensor<writeonly:tensor<7xf32>>) {
        %cst = arith.constant 0.0 : f32
        %cst1 = arith.constant 10.0 : f32
        %input = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [7, 7, 2048], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<7x7x2048xf32>> -> tensor<7x7x2048xf32>
        %init = tensor.empty() : tensor<7xf32>
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
              : tensor<7xf32> -> !flow.dispatch.tensor<writeonly:tensor<7xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 0, 0], [1, 0, 0], [0, 1, 4]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.export public @predict_dispatch_86
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic {indexing_maps = [#{{.+}}, #{{.+}}], iterator_types = ["parallel", "reduction", "reduction"]}
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_i8_i8_i32_static  {
  hal.executable.variant public @embedded_elf_x86_64, target = #hal.executable.target<
    "llvm-cpu",
    "embedded-elf-x86_64", {
      data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
      native_vector_size = 4 : index,
      target_triple = "x86_64-unknown-unknown-eabi-elf"
    }> {
    hal.executable.export public @matmul_i8_i8_i32_static layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_i8_i8_i32_static() {
        %c0_i32 = arith.constant 0 : i32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<128x384xi8>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<384x1536xi8>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x1536xi32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 384], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x384xi8>> -> tensor<128x384xi8>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [384, 1536], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<384x1536xi8>> -> tensor<384x1536xi8>
        %5 = tensor.empty() : tensor<128x1536xi32>
        %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<128x1536xi32>) -> tensor<128x1536xi32>
        %7 = linalg.matmul ins(%3, %4 : tensor<128x384xi8>, tensor<384x1536xi8>) outs(%6 : tensor<128x1536xi32>) -> tensor<128x1536xi32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 1536], strides = [1, 1] : tensor<128x1536xi32> -> !flow.dispatch.tensor<writeonly:tensor<128x1536xi32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[128, 128, 0], [8, 32, 0], [0, 0, 16]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPadExpert>
//      CHECK: hal.executable.export public @matmul_i8_i8_i32_static
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 4, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<
  "llvm-cpu", "embedded-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-unknown-eabi-elf"
  }
>
#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0, s1] -> (s1, -d0 + s0)>
hal.executable private @gemm_unit_N {
  hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.export public @gemm_unit_N ordinal(0) layout(#pipeline_layout)
    builtin.module  {
      func.func @gemm_unit_N() {
        %c0 = arith.constant 0 : index
        %M = hal.interface.constant.load[0] : index
        %K = hal.interface.constant.load[1] : index
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) offset(%c0)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%M, %K}
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32) offset(%c0)
            : !flow.dispatch.tensor<readonly:tensor<?x1xf32>>{%K}
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32) offset(%c0)
            : !flow.dispatch.tensor<readwrite:tensor<?x1xf32>>{%M}
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [%K, 1], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x1xf32>>{%K} -> tensor<?x1xf32>
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%M, %K} -> tensor<?x?xf32>
        %init = flow.dispatch.tensor.load %result_binding, offsets = [0, 0], sizes = [%M, 1], strides = [1, 1]
            : !flow.dispatch.tensor<readwrite:tensor<?x1xf32>>{%M} -> tensor<?x1xf32>
        %gemm = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x1xf32>) outs(%init : tensor<?x1xf32>) -> tensor<?x1xf32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [%M, 1], strides = [1, 1]
            : tensor<?x1xf32> -> !flow.dispatch.tensor<readwrite:tensor<?x1xf32>>{%M}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[192, 0, 0], [8, 32, 0], [0, 0, 16]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPadExpert>
//      CHECK: hal.executable.export public @gemm_unit_N
// CHECK-SAME:       translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 4, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<
  "llvm-cpu", "embedded-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-unknown-eabi-elf"
  }
>
hal.executable private @gemm_unit_M_unit_N {
  hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.export public @gemm_unit_M_unit_N ordinal(0) layout(#pipeline_layout)
    builtin.module  {
      func.func @gemm_unit_M_unit_N() {
        %c0 = arith.constant 0 : index
        %K = hal.interface.constant.load[0] : index
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x?xf32>>{%K}
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?x1xf32>>{%K}
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<1x1xf32>>
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [1, %K], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1x?xf32>>{%K} -> tensor<1x?xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [%K, 1], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x1xf32>>{%K} -> tensor<?x1xf32>
        %init = flow.dispatch.tensor.load %result_binding, offsets = [0, 0], sizes = [1, 1], strides = [1, 1]
            : !flow.dispatch.tensor<readwrite:tensor<1x1xf32>> -> tensor<1x1xf32>
        %gemm = linalg.matmul ins(%lhs, %rhs : tensor<1x?xf32>, tensor<?x1xf32>) outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [1, 1], strides = [1, 1]
            : tensor<1x1xf32> -> !flow.dispatch.tensor<readwrite:tensor<1x1xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 0, 0], [8, 32, 0], [0, 0, 16]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPadExpert>
//      CHECK: hal.executable.export public @gemm_unit_M_unit_N
// CHECK-SAME:       translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 4, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<
  "llvm-cpu", "embedded-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-unknown-eabi-elf"
  }
>
hal.executable private @matmul_odd {
  hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.export public @matmul_odd ordinal(0) layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_odd() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<33x16xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x49xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<33x49xf32>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<33x49xf32>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [33, 16], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<33x16xf32>> -> tensor<33x16xf32>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 49], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16x49xf32>> -> tensor<16x49xf32>
        %6 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [33, 49], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<33x49xf32>> -> tensor<33x49xf32>
        %7 = tensor.empty() : tensor<33x49xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<33x49xf32>) -> tensor<33x49xf32>
        %9 = linalg.matmul ins(%4, %5 : tensor<33x16xf32>, tensor<16x49xf32>) outs(%8 : tensor<33x49xf32>) -> tensor<33x49xf32>
        flow.dispatch.tensor.store %9, %3, offsets = [0, 0], sizes = [33, 49], strides = [1, 1] : tensor<33x49xf32> -> !flow.dispatch.tensor<writeonly:tensor<33x49xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[11, 32, 0], [8, 32, 0], [0, 0, 16]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPadExpert>
//      CHECK: hal.executable.export public @matmul_odd
// CHECK-SAME:       translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @generic_unit_dims_dynamic {
  hal.executable.variant @llvm, target = <"llvm-cpu", "embedded-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }> {
    hal.executable.export @generic_unit_dims_dynamic layout(#pipeline_layout)
    builtin.module {
      func.func @generic_unit_dims_dynamic() {
        %c0 = arith.constant 0 : index
        %d0 = hal.interface.constant.load[0] : index
        %d1 = hal.interface.constant.load[1] : index
        %d2 = hal.interface.constant.load[2] : index
        %d3 = hal.interface.constant.load[3] : index
        %in_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<1x?x1x1x?x?x1x?xf32>>{%d0, %d1, %d2, %d3}
        %result_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<1x?x1x1x?x?x1x?xf32>>{%d0, %d1, %d2, %d3}
        %in = flow.dispatch.tensor.load %in_binding, offsets=[0, 0, 0, 0, 0, 0, 0, 0],
            sizes=[1, %d0, 1, 1, %d1, %d2, 1, %d3], strides=[1, 1, 1, 1, 1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1x?x1x1x?x?x1x?xf32>>{%d0, %d1, %d2, %d3} -> tensor<1x?x1x1x?x?x1x?xf32>
        %init = tensor.empty(%d0, %d1, %d2, %d3) : tensor<1x?x1x1x?x?x1x?xf32>
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
            : tensor<1x?x1x1x?x?x1x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x?x1x1x?x?x1x?xf32>>{%d0, %d1, %d2, %d3}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 0, 0, 0, 64, 64, 0, 64], [1, 1, 1, 1, 1, 1, 1, 4], [0, 0, 0, 0, 0, 0, 0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPeelingExpert>
//      CHECK: hal.executable.export public @generic_unit_dims_dynamic
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @reduce_to_scalar_static {
  hal.executable.variant @llvm, target = <"llvm-cpu", "embedded-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }> {
    hal.executable.export @reduce_to_scalar_static layout(#pipeline_layout)
    builtin.module {
      func.func @reduce_to_scalar_static() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<128xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<f32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [128], strides = [1] : !flow.dispatch.tensor<readonly:tensor<128xf32>> -> tensor<128xf32>
        %3 = tensor.empty() : tensor<f32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<f32>) -> tensor<f32>
        %5 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%2 : tensor<128xf32>) outs(%4 : tensor<f32>) {
        ^bb0(%arg0: f32, %arg1: f32):
          %6 = arith.addf %arg0, %arg1 : f32
          linalg.yield %6 : f32
        } -> tensor<f32>
        flow.dispatch.tensor.store %5, %1, offsets = [], sizes = [], strides = [] : tensor<f32> -> !flow.dispatch.tensor<writeonly:tensor<f32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0], [0], [4]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.export public @reduce_to_scalar_static
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @reduce_to_scalar_dynamic {
  hal.executable.variant @llvm, target = <"llvm-cpu", "embedded-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }> {
    hal.executable.export @reduce_to_scalar_dynamic layout(#pipeline_layout)
    builtin.module {
      func.func @reduce_to_scalar_dynamic() {
        %c0 = arith.constant 0 : index
        %d0 = hal.interface.constant.load[0] : index
        %in_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%d0}
        %out_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<f32>>
        %in = flow.dispatch.tensor.load %in_binding, offsets=[0], sizes=[%d0], strides=[1] : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%d0} -> tensor<?xf32>
        %out = flow.dispatch.tensor.load %out_binding, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readwrite:tensor<f32>> -> tensor<f32>
        %reduce = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
          iterator_types = ["reduction"]}
          ins(%in : tensor<?xf32>) outs(%out : tensor<f32>) {
          ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
            %7 = arith.addf %arg0, %arg1 : f32
            linalg.yield %7 : f32
          } -> tensor<f32>
        flow.dispatch.tensor.store %reduce, %out_binding, offsets = [], sizes = [], strides = [] : tensor<f32> -> !flow.dispatch.tensor<readwrite:tensor<f32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0], [0], [4]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPeelingExpert>
//      CHECK: hal.executable.export public @reduce_to_scalar_dynamic
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @scalar {
  hal.executable.variant @llvm, target = <"llvm-cpu", "embedded-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }> {
    hal.executable.export @scalar layout(#pipeline_layout)
    builtin.module {
      func.func @scalar() {
        %c0 = arith.constant 0 : index
        %in_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<f32>>
        %out_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<f32>>
        %in = flow.dispatch.tensor.load %in_binding, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:tensor<f32>> -> tensor<f32>
        %out = flow.dispatch.tensor.load %out_binding, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<writeonly:tensor<f32>> -> tensor<f32>
        %reduce = linalg.generic {
          indexing_maps = [affine_map<() -> ()>,
                           affine_map<() -> ()>],
          iterator_types = []}
          ins(%in : tensor<f32>) outs(%out : tensor<f32>) {
          ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
            %7 = arith.addf %arg0, %arg1 : f32
            linalg.yield %7 : f32
          } -> tensor<f32>
        flow.dispatch.tensor.store %reduce, %out_binding, offsets = [], sizes = [], strides = [] : tensor<f32> -> !flow.dispatch.tensor<writeonly:tensor<f32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDefault>
//      CHECK: hal.executable.export public @scalar
// CHECK-SAME:     translation_info = #[[TRANSLATION]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<
  "llvm-cpu",
  "embedded-elf-x86_64", {
    cpu_features = "+avx2",
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 64 : index,
    target_triple = "x86_64-unknown-unknown-eabi-elf"
  }>


hal.executable private @transpose_8x8 {
  hal.executable.variant @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.export @transpose_8x8 layout(#pipeline_layout)
    builtin.module {
      func.func @transpose_8x8() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<512x1024xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1024x512xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x1024xf32>> -> tensor<512x1024xf32>
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<1024x512xf32>> -> tensor<1024x512xf32>
        %6 = linalg.generic {
          indexing_maps = [ affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%3 : tensor<512x1024xf32>) outs(%5 : tensor<1024x512xf32>) {
          ^bb0(%arg1: f32, %arg2: f32):
            linalg.yield %arg1 : f32
          } -> tensor<1024x512xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : tensor<1024x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024x512xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64], [8, 8], []{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>

// -----

hal.executable private @multi_root {
  hal.executable.variant public @embedded_elf_x86_64, target = <"llvm-cpu", "embedded-elf-x86_64", {
      cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
      native_vector_size = 16 : index, target_triple = "x86_64-unknown-unknown-eabi-elf"}> {
    hal.executable.export public @multi_root ordinal(0)
        layout(#hal.pipeline.layout<
            push_constants = 0,
            sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @multi_root() {
        %c0 = arith.constant 0 : index
        %c6144 = arith.constant 6144 : index
        %c792576 = arith.constant 792576 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
            : !flow.dispatch.tensor<readonly:tensor<12x128x128xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
            : !flow.dispatch.tensor<readonly:tensor<12x128xf32>>
        %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c792576)
            : !flow.dispatch.tensor<writeonly:tensor<12x128xf32>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [12, 128, 128], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<12x128x128xf32>> -> tensor<12x128x128xf32>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [12, 128], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<12x128xf32>> -> tensor<12x128xf32>
        %7 = tensor.empty() : tensor<12x128xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<12x128xf32>) -> tensor<12x128xf32>
        %9 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel", "reduction"]}
            ins(%4 : tensor<12x128x128xf32>) outs(%5 : tensor<12x128xf32>) {
        ^bb0(%arg0: f32, %arg1: f32):
          %11 = arith.maxf %arg0, %arg1 : f32
          linalg.yield %11 : f32
        } -> tensor<12x128xf32>
        %10 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>,
                             affine_map<(d0, d1, d2) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel", "reduction"]}
            ins(%4, %9 : tensor<12x128x128xf32>, tensor<12x128xf32>)
            outs(%8 : tensor<12x128xf32>) {
        ^bb0(%arg0: f32, %arg1: f32, %arg3: f32):
          %11 = arith.subf %arg0, %arg1 : f32
          %12 = math.exp %11 : f32
          %13 = arith.addf %12, %arg3 : f32
          linalg.yield %13 : f32
        } -> tensor<12x128xf32>
        flow.dispatch.tensor.store %10, %3, offsets = [0, 0], sizes = [12, 128], strides = [1, 1]
            : tensor<12x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<12x128xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[4, 32, 0], [1, 4, 0], [0, 0, 4]{{\]}}
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.export public @multi_root
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic
//  CHECK-NOT: lowering_config
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @pack  {
  hal.executable.variant public @embedded_elf_x86_64, target = <"llvm-cpu", "embedded-elf-x86_64", {
    cpu_features = "+avx512f",
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-unknown-eabi-elf"
  }> {
  hal.executable.export public @pack layout(#pipeline_layout)
    builtin.module {
      func.func @pack() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<20x40xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x48x16x1xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [20, 40], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<20x40xf32>> -> tensor<20x40xf32>
        %3 = tensor.empty() : tensor<2x48x16x1xf32>
        %4 = iree_linalg_ext.pack %2 padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %3 : (tensor<20x40xf32> tensor<2x48x16x1xf32>) -> tensor<2x48x16x1xf32>
        flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [2, 48, 16, 1], strides = [1, 1, 1, 1] : tensor<2x48x16x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x48x16x1xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[4, 64]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDataTiling>
//      CHECK: hal.executable.export public @pack
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   iree_linalg_ext.pack
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

hal.executable private @quant_model {
  hal.executable.variant public @embedded_elf_x86_64, target = <"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-unknown-eabi-elf"}> {
    hal.executable.export public @quant_model ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer, ReadOnly>, <4, storage_buffer, ReadOnly>, <5, storage_buffer, ReadOnly>, <6, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @quant_model() {
        %c0 = arith.constant 0 : index
        %c12_i32 = arith.constant 12 : i32
        %c-128_i32 = arith.constant -128 : i32
        %c127_i32 = arith.constant 127 : i32
        %c0_i32 = arith.constant 0 : i32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2304x24xi8>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<24x144xi8>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<144xi32>>
        %6 = hal.interface.binding.subspan set(0) binding(6) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2304x144xi8>>
        %7 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2304, 24], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2304x24xi8>> -> tensor<2304x24xi8>
        %8 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [24, 144], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<24x144xi8>> -> tensor<24x144xi8>
        %9 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [144], strides = [1] : !flow.dispatch.tensor<readonly:tensor<144xi32>> -> tensor<144xi32>
        %13 = tensor.empty() : tensor<2304x144xi8>
        %14 = tensor.empty() : tensor<2304x144xi32>
        %15 = linalg.fill ins(%c0_i32 : i32) outs(%14 : tensor<2304x144xi32>) -> tensor<2304x144xi32>
        %16 = linalg.matmul ins(%7, %8 : tensor<2304x24xi8>, tensor<24x144xi8>) outs(%15 : tensor<2304x144xi32>) -> tensor<2304x144xi32>
        %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9, %16 : tensor<144xi32>, tensor<2304x144xi32>) outs(%13 : tensor<2304x144xi8>) {
        ^bb0(%in: i32, %in_0: i32, %out: i8):
          %19 = arith.subi %in_0, %c12_i32 : i32
          %20 = arith.addi %in, %19 : i32
          %27 = arith.trunci %20 : i32 to i8
          linalg.yield %27 : i8
        } -> tensor<2304x144xi8>
        flow.dispatch.tensor.store %17, %6, offsets = [0, 0], sizes = [2304, 144], strides = [1, 1] : tensor<2304x144xi8> -> !flow.dispatch.tensor<writeonly:tensor<2304x144xi8>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[256, 72, 0], [8, 32, 0], [0, 0, 12]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPadExpert>
//      CHECK: hal.executable.export public @quant_model
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]
