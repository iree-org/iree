// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target-pass{test-lowering-configuration}))' %s | FileCheck %s

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @add_dispatch_0 {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
  hal.executable.entry_point @add_dispatch_0 layout(#executable_layout)
  builtin.module {
    func @add_dispatch_0() {
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:16384xf32>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:16384xf32>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:16384xf32>
      %3 = linalg.init_tensor [16384] : tensor<16384xf32>
      %4 = flow.dispatch.tensor.load %0, offsets=[0], sizes=[16384], strides=[1] : !flow.dispatch.tensor<readonly:16384xf32> -> tensor<16384xf32>
      %5 = flow.dispatch.tensor.load %1, offsets=[0], sizes=[16384], strides=[1] : !flow.dispatch.tensor<readonly:16384xf32> -> tensor<16384xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4, %5 : tensor<16384xf32>, tensor<16384xf32>) outs(%3 : tensor<16384xf32>) {
      ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
          %7 = arith.addf %arg0, %arg1 : f32
          linalg.yield %7 : f32
        } -> tensor<16384xf32>
        flow.dispatch.tensor.store %6, %2, offsets=[0], sizes=[16384], strides=[1] : tensor<16384xf32> -> !flow.dispatch.tensor<writeonly:16384xf32>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[256]{{\]}}, native_vector_size = []>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"LLVMGPUVectorize", workload_per_wg = []>
//      CHECK: hal.executable.entry_point public @add_dispatch_0
// CHECK-SAME:     translation.info = #[[TRANSLATION]]
// CHECK-SAME:     workgroup_size = [64 : index, 1 : index, 1 : index]
//      CHECK: func @add_dispatch_0
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering.config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @dot_dispatch_1  {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @dot_dispatch_1 layout(#executable_layout)
    builtin.module {
      func @dot_dispatch_1() {
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c2 = arith.constant 2 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<2x3xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<3x4xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<2x4xf32>
              linalg.fill(%cst, %2) : f32, memref<2x4xf32>
              linalg.matmul ins(%0, %1 : memref<2x3xf32>, memref<3x4xf32>) outs(%2 : memref<2x4xf32>)
              return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[4, 2, 4]{{\]}}, native_vector_size = []>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"LLVMGPUMatmulSimt", workload_per_wg = []>
//      CHECK: hal.executable.entry_point public @dot_dispatch_1
// CHECK-SAME:     translation.info = #[[TRANSLATION]]
// CHECK-SAME:     workgroup_size = [2 : index, 4 : index, 1 : index]
//      CHECK: func @dot_dispatch_1
//      CHECK:   linalg.fill
// CHECK-SAME:       lowering.config = #[[CONFIG]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering.config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @reduction_dispatch {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @predict_dispatch_153 layout(#executable_layout)
    builtin.module {
      func @predict_dispatch_153() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0x7FC00000 : f32
        %cst_0 = arith.constant 0xFF800000 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<1000xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<f32>
        linalg.fill(%cst_0, %1) : f32, memref<f32>
        linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%0 : memref<1000xf32>) outs(%1 : memref<f32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %2 = arith.cmpf ogt, %arg0, %arg1 : f32
          %3 = arith.select %2, %arg0, %arg1 : f32
          %4 = arith.cmpf uno, %arg0, %arg1 : f32
          %5 = arith.select %4, %cst, %3 : f32
          linalg.yield %5 : f32
        }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[]{{\]}}, native_vector_size = []>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"LLVMGPUDistribute", workload_per_wg = []>
//      CHECK: hal.executable.entry_point public @predict_dispatch_153
// CHECK-SAME:     translation.info = #[[TRANSLATION]]
//      CHECK: linalg.fill
// CHECK-SAME:   lowering.config = #[[CONFIG]]
//      CHECK: linalg.generic
// CHECK-SAME:   lowering.config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @tensor_insert_slice {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @tensor_insert_slice layout(#executable_layout)
    builtin.module {
      builtin.func @tensor_insert_slice() {
        %c0 = arith.constant 0 : index
        %size_y = hal.interface.constant.load[0] : index
        %size_x = hal.interface.constant.load[1] : index
        %dest_size_y = hal.interface.constant.load[2] : index
        %dest_size_x = hal.interface.constant.load[3] : index
        %offset_y = hal.interface.constant.load[4] : index
        %offset_x = hal.interface.constant.load[5] : index
        %source_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:?x?xi32>{%size_y, %size_x}
        %dest_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readwrite:?x?xi32>{%dest_size_y, %dest_size_x}
        %source = flow.dispatch.tensor.load %source_binding, offsets = [0, 0], sizes = [%size_y, %size_x], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:?x?xi32>{%size_y, %size_x} -> tensor<?x?xi32>
        %dest = flow.dispatch.tensor.load %dest_binding, offsets = [0, 0], sizes = [%dest_size_y, %dest_size_x], strides = [1, 1]
            : !flow.dispatch.tensor<readwrite:?x?xi32>{%dest_size_y, %dest_size_x} -> tensor<?x?xi32>
        %result = tensor.insert_slice %source into %dest[%offset_y, %offset_x] [%size_y, %size_x] [1, 1]
            : tensor<?x?xi32> into tensor<?x?xi32>
        flow.dispatch.tensor.store %result, %dest_binding, offsets = [0, 0], sizes = [%dest_size_y, %dest_size_x], strides = [1, 1]
            : tensor<?x?xi32> -> !flow.dispatch.tensor<readwrite:?x?xi32>{%dest_size_y, %dest_size_x}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[1, 64]{{\]}}, native_vector_size = []>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"LLVMGPUVectorize", workload_per_wg = []>
//      CHECK: hal.executable.entry_point public @tensor_insert_slice
// CHECK-SAME:     translation.info = #[[TRANSLATION]]
//      CHECK: tensor.insert_slice
// CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @copy_as_generic {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @copy_as_generic layout(#executable_layout)
    builtin.module {
      builtin.func @copy_as_generic() {
        %c0 = arith.constant 0 : index
        %d0 = hal.interface.constant.load[0] : index
        %d1 = hal.interface.constant.load[1] : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<?x?xi32>{%d0, %d1}
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<?x?xi32>{%d0, %d1}
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
            ins(%0 : memref<?x?xi32>) outs(%1 : memref<?x?xi32>) {
          ^bb0(%arg4: i32, %s: i32):  // no predecessors
            linalg.yield %arg4 : i32
          }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[1, 64]{{\]}}, native_vector_size = []>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"LLVMGPUVectorize", workload_per_wg = []>
//      CHECK: hal.executable.entry_point public @copy_as_generic
// CHECK-SAME:     translation.info = #[[TRANSLATION]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @static_1d_fft_stage2 {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
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

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[4]{{\]}}, native_vector_size = []>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"LLVMGPUDistribute", workload_per_wg = []>
//       CHECK: hal.executable.entry_point public @static_1d_fft_stage2
//  CHECK-SAME:     translation.info = #[[TRANSLATION]]
//       CHECK: iree_linalg_ext.fft
//  CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @static_3d_fft_stage3 {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @static_3d_fft_stage3 layout(#executable_layout)
    builtin.module {
      builtin.func @static_3d_fft_stage3() {
        %c0 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c64 = arith.constant 64 : index
        %c128 = arith.constant 128 : index
        %c32 = arith.constant 32 : index
        %cst = arith.constant dense<[1.000000e+00, 0.707106769, 6.12323426E-17, -0.707106769]> : tensor<4xf32>
        %cst_0 = arith.constant dense<[-0.000000e+00, -0.707106769, -1.000000e+00, -0.707106769]> : tensor<4xf32>
        %0 = bufferization.to_memref %cst_0 : memref<4xf32>
        %1 = bufferization.to_memref %cst : memref<4xf32>
        %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<64x128x32xf32>
        %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<64x128x32xf32>
        iree_linalg_ext.fft {__internal_linalg_transform__ = "workgroup"}
            ins(%c3, %1, %0 : index, memref<4xf32>, memref<4xf32>)
            outs(%2, %3 : memref<64x128x32xf32>, memref<64x128x32xf32>)
        return
      }
    }
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[1, 1, 8]{{\]}}, native_vector_size = []>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"LLVMGPUDistribute", workload_per_wg = []>
//       CHECK: hal.executable.entry_point public @static_3d_fft_stage3
//  CHECK-SAME:     translation.info = #[[TRANSLATION]]
//       CHECK: iree_linalg_ext.fft
//  CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

#compilation = #iree_codegen.compilation.info<
    #iree_codegen.lowering.config<tile_sizes = [[32, 128, 64]], native_vector_size = []>,
    #iree_codegen.translation.info<"LLVMGPUMatmulSimt", workload_per_wg = [256, 32]>,
    workgroup_size = [16, 8, 1]>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @user_config {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb"> {
  hal.executable.entry_point public @_lowering_config_test_dispatch_1 layout(#executable_layout)
  builtin.module {
    func @_lowering_config_test_dispatch_1() {
      %cst = arith.constant 0.000000e+00 : f32
      %c128 = arith.constant 128 : index
      %c1024 = arith.constant 1024 : index
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:128x256xf32>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:256x1024xf32>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:128x1024xf32>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:128x256xf32> -> tensor<128x256xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:256x1024xf32> -> tensor<256x1024xf32>
      %15 = linalg.init_tensor [128, 1024] : tensor<128x1024xf32>
      %16 = linalg.fill(%cst, %15) : f32, tensor<128x1024xf32> -> tensor<128x1024xf32>
      %17 = linalg.matmul {__internal_linalg_transform__ = "workgroup", compilation.info = #compilation}
          ins(%3, %4 : tensor<128x256xf32>, tensor<256x1024xf32>) outs(%16 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
      flow.dispatch.tensor.store %17, %2, offsets = [0, 0], sizes = [128, 1024], strides = [1, 1] : tensor<128x1024xf32> -> !flow.dispatch.tensor<writeonly:128x1024xf32>
      return
    }
  }
}
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[32, 128, 64]{{\]}}
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"LLVMGPUMatmulSimt", workload_per_wg = [256, 32]>
//      CHECK: hal.executable.entry_point public @_lowering_config_test_dispatch_1
// CHECK-SAME:     translation.info = #[[TRANSLATION]]
// CHECK-SAME:     workgroup_size = [16 : index, 8 : index, 1 : index]
//      CHECK: linalg.fill
// CHECK-SAME:     lowering.config = #[[CONFIG]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
hal.executable private @sort_op {
  hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}> {
    hal.executable.entry_point public @sort_op layout(#executable_layout)
    builtin.module {
      func @sort_op() {
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %c2304000 = arith.constant 2304000 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) : !flow.dispatch.tensor<readonly:1x576000xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32) : !flow.dispatch.tensor<readonly:1x576000xi32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32) : !flow.dispatch.tensor<writeonly:1x576000xf32>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) offset(%c2304000) alignment(32) : !flow.dispatch.tensor<writeonly:1x576000xi32>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 576000], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:1x576000xf32> -> tensor<1x576000xf32>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1, 576000], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:1x576000xi32> -> tensor<1x576000xi32>
        %9:2 = iree_linalg_ext.sort dimension(1) outs(%4, %5 : tensor<1x576000xf32>, tensor<1x576000xi32>)  {
        ^bb0(%arg1: f32, %arg2: f32, %arg3: i32, %arg4: i32):  // no predecessors
          %10 = arith.cmpf ogt, %arg1, %arg2 : f32
          iree_linalg_ext.yield %10 : i1
        } -> tensor<1x576000xf32>, tensor<1x576000xi32>
        flow.dispatch.tensor.store %9#0, %2, offsets = [0, 0], sizes = [1, 576000], strides = [1, 1] : tensor<1x576000xf32> -> !flow.dispatch.tensor<writeonly:1x576000xf32>
        flow.dispatch.tensor.store %9#1, %3, offsets = [0, 0], sizes = [1, 576000], strides = [1, 1] : tensor<1x576000xi32> -> !flow.dispatch.tensor<writeonly:1x576000xi32>
        return
      }
    }
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[64]{{\]}}, native_vector_size = []>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"LLVMGPUDistribute", workload_per_wg = []>
//       CHECK: hal.executable.entry_point public @sort_op
//  CHECK-SAME:     translation.info = #[[TRANSLATION]]
//       CHECK: iree_linalg_ext.sort
//  CHECK-SAME:     lowering.config = #[[CONFIG]]
