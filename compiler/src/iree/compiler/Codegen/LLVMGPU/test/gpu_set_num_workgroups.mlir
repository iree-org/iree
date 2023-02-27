// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target{test-lowering-configuration})))" \
// RUN: --iree-codegen-llvmgpu-enable-transform-dialect-jit=false %s | FileCheck %s
// Transform dialect attributes are tested separately.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @add_dispatch_0 {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
  hal.executable.export @add_dispatch_0 layout(#pipeline_layout)
  builtin.module {
    func.func @add_dispatch_0() {
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<16384xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<16384xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<16384xf32>>
      %3 = tensor.empty() : tensor<16384xf32>
      %4 = flow.dispatch.tensor.load %0, offsets=[0], sizes=[16384], strides=[1] : !flow.dispatch.tensor<readonly:tensor<16384xf32>> -> tensor<16384xf32>
      %5 = flow.dispatch.tensor.load %1, offsets=[0], sizes=[16384], strides=[1] : !flow.dispatch.tensor<readonly:tensor<16384xf32>> -> tensor<16384xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4, %5 : tensor<16384xf32>, tensor<16384xf32>) outs(%3 : tensor<16384xf32>) {
      ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
          %7 = arith.addf %arg0, %arg1 : f32
          linalg.yield %7 : f32
        } -> tensor<16384xf32>
        flow.dispatch.tensor.store %6, %2, offsets=[0], sizes=[16384], strides=[1] : tensor<16384xf32> -> !flow.dispatch.tensor<writeonly:tensor<16384xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[256]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUVectorize>
//      CHECK: hal.executable.export public @add_dispatch_0
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
// CHECK-SAME:     workgroup_size = [64 : index, 1 : index, 1 : index]
//      CHECK: func.func @add_dispatch_0
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
hal.executable private @dot_dispatch_1  {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.export @dot_dispatch_1 layout(#pipeline_layout)
    builtin.module {
      func.func @dot_dispatch_1() {
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c2 = arith.constant 2 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<2x3xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<3x4xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<2x4xf32>
        linalg.fill ins(%cst : f32) outs(%2 : memref<2x4xf32>)
        linalg.matmul ins(%0, %1 : memref<2x3xf32>, memref<3x4xf32>) outs(%2 : memref<2x4xf32>)
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[4, 2, 4]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUMatmulSimt>
//      CHECK: hal.executable.export public @dot_dispatch_1
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
// CHECK-SAME:     workgroup_size = [2 : index, 4 : index, 1 : index]
//      CHECK: func.func @dot_dispatch_1
//      CHECK:   linalg.fill
// CHECK-SAME:       lowering_config = #[[CONFIG]]
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
hal.executable private @unaligned_k  {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.export @unaligned_k layout(#pipeline_layout)
    builtin.module {
      func.func @unaligned_k() {
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c2 = arith.constant 2 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<128x258xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<258x64xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<128x64xf32>
        linalg.fill ins(%cst : f32) outs(%2 : memref<128x64xf32>)
        linalg.matmul ins(%0, %1 : memref<128x258xf32>, memref<258x64xf32>) outs(%2 : memref<128x64xf32>)
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32, 128, 2]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUMatmulSimt>
//      CHECK: hal.executable.export public @unaligned_k
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
// CHECK-SAME:     workgroup_size = [32 : index, 8 : index, 1 : index]
//      CHECK: func.func @unaligned_k
//      CHECK:   linalg.fill
// CHECK-SAME:       lowering_config = #[[CONFIG]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]


// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @reduction_dispatch {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.export @predict_dispatch_153 layout(#pipeline_layout)
    builtin.module {
      func.func @predict_dispatch_153() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0x7FC00000 : f32
        %cst_0 = arith.constant 0xFF800000 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<1000xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<f32>
        linalg.fill ins(%cst_0 : f32) outs(%1 : memref<f32>)
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

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUDistribute>
//      CHECK: hal.executable.export public @predict_dispatch_153
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.fill
// CHECK-SAME:   lowering_config = #[[CONFIG]]
//      CHECK: linalg.generic
// CHECK-SAME:   lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @reduction_aligned2 {
  hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.export public @reduction_aligned2 ordinal(0) layout(#pipeline_layout)
    builtin.module {
      func.func @reduction_aligned2() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<4x128x384xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x384xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [4, 128, 384], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x128x384xf32>> -> tensor<4x128x384xf32>
        %3 = tensor.empty() : tensor<128x384xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<128x384xf32>) -> tensor<128x384xf32>
        %5 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0, d1)>,
                           affine_map<(d0, d1, d2) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel", "reduction"]}
          ins(%2 : tensor<4x128x384xf32>) outs(%4 : tensor<128x384xf32>) {
        ^bb0(%arg0: f32, %arg1: f32):
          %6 = arith.addf %arg0, %arg1 : f32
          linalg.yield %6 : f32
        } -> tensor<128x384xf32>
        flow.dispatch.tensor.store %5, %1, offsets = [0, 0], sizes = [128, 384], strides = [1, 1] : tensor<128x384xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x384xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 128, 4]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUVectorize>
//      CHECK: hal.executable.export public @reduction_aligned2
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.fill
// CHECK-SAME:   lowering_config = #[[CONFIG]]
//      CHECK: linalg.generic
// CHECK-SAME:   lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @copy_as_generic {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.export @copy_as_generic layout(#pipeline_layout)
    builtin.module {
      func.func @copy_as_generic() {
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

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 64]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUDistribute>
//      CHECK: hal.executable.export public @copy_as_generic
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @static_1d_fft_stage2 {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
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

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[4]{{\]}}>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUDistribute>
//       CHECK: hal.executable.export public @static_1d_fft_stage2
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: iree_linalg_ext.fft
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @static_3d_fft_stage3 {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.export @static_3d_fft_stage3 layout(#pipeline_layout)
    builtin.module {
      func.func @static_3d_fft_stage3() {
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

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1, 8]{{\]}}>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUDistribute>
//       CHECK: hal.executable.export public @static_3d_fft_stage3
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: iree_linalg_ext.fft
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#compilation = #iree_codegen.compilation_info<
    lowering_config  = <tile_sizes = [[32, 128, 64]]>,
    translation_info = <LLVMGPUMatmulSimt>,
    workgroup_size = [16, 8, 1]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @user_config {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb"> {
  hal.executable.export public @_lowering_config_test_dispatch_1 layout(#pipeline_layout)
  builtin.module {
    func.func @_lowering_config_test_dispatch_1() {
      %cst = arith.constant 0.000000e+00 : f32
      %c128 = arith.constant 128 : index
      %c1024 = arith.constant 1024 : index
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<128x256xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<256x1024xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<128x1024xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:tensor<128x256xf32>> -> tensor<128x256xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:tensor<256x1024xf32>> -> tensor<256x1024xf32>
      %15 = tensor.empty() : tensor<128x1024xf32>
      %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
      %17 = linalg.matmul {__internal_linalg_transform__ = "workgroup", compilation_info = #compilation}
          ins(%3, %4 : tensor<128x256xf32>, tensor<256x1024xf32>) outs(%16 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
      flow.dispatch.tensor.store %17, %2, offsets = [0, 0], sizes = [128, 1024], strides = [1, 1] : tensor<128x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x1024xf32>>
      return
    }
  }
}
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32, 128, 64]{{\]}}
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUMatmulSimt>
//      CHECK: hal.executable.export public @_lowering_config_test_dispatch_1
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
// CHECK-SAME:     workgroup_size = [16 : index, 8 : index, 1 : index]
//      CHECK: linalg.fill
// CHECK-SAME:     lowering_config = #[[CONFIG]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
hal.executable private @sort_op {
  hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}> {
    hal.executable.export public @sort_op layout(#pipeline_layout)
    builtin.module {
      func.func @sort_op() {
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %c2304000 = arith.constant 2304000 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) : !flow.dispatch.tensor<readonly:tensor<1x576000xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32) : !flow.dispatch.tensor<readonly:tensor<1x576000xi32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32) : !flow.dispatch.tensor<writeonly:tensor<1x576000xf32>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(32) offset(%c2304000) : !flow.dispatch.tensor<writeonly:tensor<1x576000xi32>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 576000], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1x576000xf32>> -> tensor<1x576000xf32>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1, 576000], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1x576000xi32>> -> tensor<1x576000xi32>
        %9:2 = iree_linalg_ext.sort dimension(1) outs(%4, %5 : tensor<1x576000xf32>, tensor<1x576000xi32>)  {
        ^bb0(%arg1: f32, %arg2: f32, %arg3: i32, %arg4: i32):  // no predecessors
          %10 = arith.cmpf ogt, %arg1, %arg2 : f32
          iree_linalg_ext.yield %10 : i1
        } -> tensor<1x576000xf32>, tensor<1x576000xi32>
        flow.dispatch.tensor.store %9#0, %2, offsets = [0, 0], sizes = [1, 576000], strides = [1, 1] : tensor<1x576000xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x576000xf32>>
        flow.dispatch.tensor.store %9#1, %3, offsets = [0, 0], sizes = [1, 576000], strides = [1, 1] : tensor<1x576000xi32> -> !flow.dispatch.tensor<writeonly:tensor<1x576000xi32>>
        return
      }
    }
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64]{{\]}}>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUDistribute>
//       CHECK: hal.executable.export public @sort_op
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: iree_linalg_ext.sort
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @user_config {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}> {
  hal.executable.export public @matmul_config_sm35 layout(#pipeline_layout)
  builtin.module {
    func.func @matmul_config_sm35() {
      %cst = arith.constant 0.000000e+00 : f32
      %c128 = arith.constant 128 : index
      %c1024 = arith.constant 1024 : index
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<128x256xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<256x1024xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<128x1024xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:tensor<128x256xf32>> -> tensor<128x256xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:tensor<256x1024xf32>> -> tensor<256x1024xf32>
      %15 = tensor.empty() : tensor<128x1024xf32>
      %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
      %17 = linalg.matmul
          ins(%3, %4 : tensor<128x256xf32>, tensor<256x1024xf32>) outs(%16 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
      flow.dispatch.tensor.store %17, %2, offsets = [0, 0], sizes = [128, 1024], strides = [1, 1] : tensor<128x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x1024xf32>>
      return
    }
  }
}
}

//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUMatmulSimt>
//      CHECK: hal.executable.export public @matmul_config_sm35
// CHECK-SAME:     translation_info = #[[TRANSLATION]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @user_config {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}> {
  hal.executable.export public @matmul_config_sm80 layout(#pipeline_layout)
  builtin.module {
    func.func @matmul_config_sm80() {
      %cst = arith.constant 0.000000e+00 : f32
      %c128 = arith.constant 128 : index
      %c1024 = arith.constant 1024 : index
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<128x256xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<256x1024xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<128x1024xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:tensor<128x256xf32>> -> tensor<128x256xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:tensor<256x1024xf32>> -> tensor<256x1024xf32>
      %15 = tensor.empty() : tensor<128x1024xf32>
      %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
      %17 = linalg.matmul
          ins(%3, %4 : tensor<128x256xf32>, tensor<256x1024xf32>) outs(%16 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
      flow.dispatch.tensor.store %17, %2, offsets = [0, 0], sizes = [128, 1024], strides = [1, 1] : tensor<128x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x1024xf32>>
      return
    }
  }
}
}

//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUMatmulTensorCore
//      CHECK: hal.executable.export public @matmul_config_sm80
// CHECK-SAME:     translation_info = #[[TRANSLATION]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @user_config {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_86"}> {
  hal.executable.export public @matmul_config_sm86 layout(#pipeline_layout)
  builtin.module {
    func.func @matmul_config_sm86() {
      %cst = arith.constant 0.000000e+00 : f32
      %c128 = arith.constant 128 : index
      %c1024 = arith.constant 1024 : index
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<128x256xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<256x1024xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<128x1024xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:tensor<128x256xf32>> -> tensor<128x256xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:tensor<256x1024xf32>> -> tensor<256x1024xf32>
      %15 = tensor.empty() : tensor<128x1024xf32>
      %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
      %17 = linalg.matmul
          ins(%3, %4 : tensor<128x256xf32>, tensor<256x1024xf32>) outs(%16 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
      flow.dispatch.tensor.store %17, %2, offsets = [0, 0], sizes = [128, 1024], strides = [1, 1] : tensor<128x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x1024xf32>>
      return
    }
  }
}
}

//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUMatmulTensorCore
//      CHECK: hal.executable.export public @matmul_config_sm86
// CHECK-SAME:     translation_info = #[[TRANSLATION]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @contract_reduction {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_86"}> {
  hal.executable.export public @contract_reduction layout(#pipeline_layout)
  builtin.module {
    func.func @contract_reduction() {
      %c0 = arith.constant 0 : index
      %c40064 = arith.constant 40064 : index
      %c34752 = arith.constant 34752 : index
      %cst = arith.constant 0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<3x7xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c40064) : !flow.dispatch.tensor<readonly:tensor<3x64x4x8xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c34752) : !flow.dispatch.tensor<writeonly:tensor<3x64xf32>>
      %3 = tensor.empty() : tensor<3x64xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 4], sizes = [3, 64, 4, 1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x64x4x8xf32>> -> tensor<3x64x4xf32>
      %5 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 128, 32]]>} ins(%cst : f32) outs(%3 : tensor<3x64xf32>) -> tensor<3x64xf32>
      %6 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<3x7xf32>> -> tensor<f32>
      %7 = linalg.generic {indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> ()>,
        affine_map<(d0, d1, d2) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%4, %6 : tensor<3x64x4xf32>, tensor<f32>) outs(%5 : tensor<3x64xf32>)  {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %8 = arith.subf %in, %in_0 : f32
        %9 = arith.maxf %8, %cst : f32
        %10 = arith.mulf %9, %9 : f32
        %11 = arith.addf %out, %10 : f32
        linalg.yield %11 : f32
      } -> tensor<3x64xf32>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [3, 64], strides = [1, 1] : tensor<3x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<3x64xf32>>
      return
    }
  }
}
}

//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUVectorize
//      CHECK: hal.executable.export public @contract_reduction
// CHECK-SAME:     translation_info = #[[TRANSLATION]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @dynamic_pack_2x2 {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_86"}> {
  hal.executable.export public @dynamic_pack_2x2 layout(#pipeline_layout)
  builtin.module {
    func.func @dynamic_pack_2x2() {
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %0 = hal.interface.constant.load[0] : i32
      %1 = hal.interface.constant.load[1] : i32
      %2 = hal.interface.constant.load[2] : i32
      %3 = hal.interface.constant.load[3] : i32
      %4 = arith.index_castui %0 : i32 to index
      %5 = arith.index_castui %1 : i32 to index
      %6 = arith.index_castui %2 : i32 to index
      %7 = arith.index_castui %3 : i32 to index
      %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c64) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x?xi32>>{%4, %5}
      %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?x?x2x2xi32>>{%6, %7}
      %10 = flow.dispatch.tensor.load %8, offsets = [0, 0], sizes = [%4, %5], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xi32>>{%4, %5} -> tensor<?x?xi32>
      %11 = tensor.empty(%6, %7) : tensor<?x?x2x2xi32>
      %pack = tensor.pack %10 inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %11 : tensor<?x?xi32> -> tensor<?x?x2x2xi32>
      flow.dispatch.tensor.store %pack, %9, offsets = [0, 0, 0, 0], sizes = [%6, %7, 2, 2], strides = [1, 1, 1, 1] : tensor<?x?x2x2xi32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x2x2xi32>>{%6, %7}
      return
    }
  }
}
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[16, 16]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUPackUnPack>
//      CHECK: hal.executable.export public @dynamic_pack_2x2
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   func.func @dynamic_pack_2x2
//      CHECK:     tensor.pack
// CHECK-SAME:       lowering_config = #[[CONFIG]]
