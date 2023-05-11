// RUN: iree-opt --split-input-file --verify-diagnostics --pass-pipeline="builtin.module(func.func(iree-flow-form-dispatch-regions{fuse-multi-use=true}, iree-flow-clone-producers-into-dispatch-regions, iree-flow-form-dispatch-workgroups), cse, canonicalize, cse)" %s | FileCheck %s

hal.executable private @matmul_alone {
  hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}> {
    hal.executable.export public @matmul_alone ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {translation_info = #iree_codegen.translation_info<LLVMGPUMatmulTensorCore pipeline_depth = 3>, workgroup_size = [128 : index, 2 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %c2048 = arith.constant 2048 : index
      %c1 = arith.constant 1 : index
      hal.return %c2048, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @matmul_alone() {
        %c128 = arith.constant 128 : index
        %c256 = arith.constant 256 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x8192xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x8192xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<8192x8192xf32>>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %3 = affine.apply affine_map<()[s0] -> ((s0 floordiv 32) * 128)>()[%workgroup_id_x]
        %4 = flow.dispatch.tensor.load %0, offsets = [%3, 0], sizes = [%c128, 8192], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x8192xf32>> -> tensor<?x8192xf32>
        %5 = affine.apply affine_map<()[s0] -> (s0 * 256 - (s0 floordiv 32) * 8192)>()[%workgroup_id_x]
        %6 = flow.dispatch.tensor.load %1, offsets = [0, %5], sizes = [8192, %c256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x8192xf32>> -> tensor<8192x?xf32>
        %7 = affine.apply affine_map<()[s0] -> ((s0 floordiv 32) * 128)>()[%workgroup_id_x]
        %8 = affine.apply affine_map<()[s0] -> (s0 * 256 - (s0 floordiv 32) * 8192)>()[%workgroup_id_x]
        %9 = flow.dispatch.tensor.load %2, offsets = [%7, %8], sizes = [%c128, %c256], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<8192x8192xf32>> -> tensor<?x?xf32>
        %10 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 256, 32]]>} ins(%4, %6 : tensor<?x8192xf32>, tensor<8192x?xf32>) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %11 = affine.apply affine_map<()[s0] -> ((s0 floordiv 32) * 128)>()[%workgroup_id_x]
        %12 = affine.apply affine_map<()[s0] -> (s0 * 256 - (s0 floordiv 32) * 8192)>()[%workgroup_id_x]
        flow.dispatch.tensor.store %10, %2, offsets = [%11, %12], sizes = [%c128, %c256], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<readwrite:tensor<8192x8192xf32>>
        return
      }
    }
  }
}

// CHECK-NOT: linalg.matmul
// CHECK: %[[cst:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[c128:.+]] = arith.constant 128 : index
// CHECK: %[[c256:.+]] = arith.constant 256 : index
// CHECK: %[[lhs:.+]] = flow.dispatch.tensor.load %{{.*}}, offsets = [%{{.*}}, 0], sizes = [%c128, 8192], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x8192xf32>> -> tensor<?x8192xf32>
// CHECK: %[[rhs:.+]] = flow.dispatch.tensor.load %{{.*}}, offsets = [0, %{{.*}}], sizes = [8192, %c256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x8192xf32>> -> tensor<8192x?xf32>
// CHECK: %[[res:.+]]  = flow.dispatch.tensor.load %{{.*}}, offsets = [%{{.*}}, %{{.*}}], sizes = [%c128, %c256], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<8192x8192xf32>> -> tensor<?x?xf32>
// CHECK: %[[OUT:.+]] = bufferization.alloc_tensor() : tensor<128x256xf32>
// CHECK: %[[REM:.+]] = bufferization.alloc_tensor() : tensor<4096xf32>
// CHECK: iree_codegen.ukernel.generic "__iree_matmul_tf32_tf32_f32_128_256_32_3_false_false" ins(%[[lhs]], %[[rhs]] : tensor<?x8192xf32>, tensor<8192x?xf32>) outs(%[[res]] : tensor<?x?xf32>) (%[[OUT]], %[[REM]], %[[cst]] : tensor<128x256xf32>, tensor<4096xf32>, f32) fn_def_attrs {} strided_outer_dims(0) -> tensor<?x?xf32> 

// -----

hal.executable private @fill_matmul {
  hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}> {
    hal.executable.export public @fill_matmul ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>]>]>) attributes {translation_info = #iree_codegen.translation_info<LLVMGPUMatmulTensorCore pipeline_depth = 4>, workgroup_size = [128 : index, 2 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %c2048 = arith.constant 2048 : index
      %c1 = arith.constant 1 : index
      hal.return %c2048, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @fill_matmul() {
        %c128 = arith.constant 128 : index
        %c256 = arith.constant 256 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 123.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<8192x8192xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x8192xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x8192xf32>>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %3 = affine.apply affine_map<()[s0] -> ((s0 floordiv 32) * 128)>()[%workgroup_id_x]
        %4 = flow.dispatch.tensor.load %1, offsets = [%3, 0], sizes = [%c128, 8192], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x8192xf32>> -> tensor<?x8192xf32>
        %5 = affine.apply affine_map<()[s0] -> (s0 * 256 - (s0 floordiv 32) * 8192)>()[%workgroup_id_x]
        %6 = flow.dispatch.tensor.load %2, offsets = [0, %5], sizes = [8192, %c256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x8192xf32>> -> tensor<8192x?xf32>
        %7 = affine.apply affine_map<()[s0] -> ((s0 floordiv 32) * 128)>()[%workgroup_id_x]
        %8 = affine.apply affine_map<()[s0] -> (s0 * 256 - (s0 floordiv 32) * 8192)>()[%workgroup_id_x]
        %9 = flow.dispatch.tensor.load %0, offsets = [%7, %8], sizes = [%c128, %c256], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<8192x8192xf32>> -> tensor<?x?xf32>
        %10 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 256, 32]]>} ins(%cst : f32) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %11 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 256, 32]]>} ins(%4, %6 : tensor<?x8192xf32>, tensor<8192x?xf32>) outs(%10 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %12 = affine.apply affine_map<()[s0] -> ((s0 floordiv 32) * 128)>()[%workgroup_id_x]
        %13 = affine.apply affine_map<()[s0] -> (s0 * 256 - (s0 floordiv 32) * 8192)>()[%workgroup_id_x]
        flow.dispatch.tensor.store %11, %0, offsets = [%12, %13], sizes = [%c128, %c256], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<readwrite:tensor<8192x8192xf32>>
        return
      }
    }
  }
}

// CHECK-NOT: linalg.fill
// CHECK-NOT: linalg.matmul

// CHECK-NOT: linalg.matmul
// CHECK: %[[c128:.+]] = arith.constant 128 : index
// CHECK: %[[c256:.+]] = arith.constant 256 : index
// CHECK: %[[cst:.+]] = arith.constant 1.230000e+02 : f32
// CHECK: %[[lhs:.+]] = flow.dispatch.tensor.load %{{.*}}, offsets = [%{{.*}}, 0], sizes = [%c128, 8192], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x8192xf32>> -> tensor<?x8192xf32>
// CHECK: %[[rhs:.+]] = flow.dispatch.tensor.load %{{.*}}, offsets = [0, %{{.*}}], sizes = [8192, %c256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x8192xf32>> -> tensor<8192x?xf32>
// CHECK: %[[res:.+]]  = flow.dispatch.tensor.load %{{.*}}, offsets = [%{{.*}}, %{{.*}}], sizes = [%c128, %c256], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<8192x8192xf32>> -> tensor<?x?xf32>
// CHECK: %[[OUT:.+]] = bufferization.alloc_tensor() : tensor<128x256xf32>
// CHECK: %[[REM:.+]] = bufferization.alloc_tensor() : tensor<16384xf32>
// CHECK: iree_codegen.ukernel.generic "__iree_matmul_tf32_tf32_f32_128_256_32_4_true_false" ins(%[[lhs]], %[[rhs]] : tensor<?x8192xf32>, tensor<8192x?xf32>) outs(%[[res]] : tensor<?x?xf32>) (%[[OUT]], %[[REM]], %[[cst]] : tensor<128x256xf32>, tensor<16384xf32>, f32)


// -----


hal.executable private @fill_matmul_generic {
  hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}> {
    hal.executable.export public @fill_matmul_generic ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>]>]>) attributes {translation_info = #iree_codegen.translation_info<LLVMGPUMatmulTensorCore pipeline_depth = 3>, workgroup_size = [128 : index, 2 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %c2048 = arith.constant 2048 : index
      %c1 = arith.constant 1 : index
      hal.return %c2048, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @fill_matmul_generic() {
        %c128 = arith.constant 128 : index
        %c256 = arith.constant 256 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x8192xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x8192xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x8192xf32>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8192x8192xf32>>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %4 = affine.apply affine_map<()[s0] -> ((s0 floordiv 32) * 128)>()[%workgroup_id_x]
        %5 = flow.dispatch.tensor.load %1, offsets = [%4, 0], sizes = [%c128, 8192], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x8192xf32>> -> tensor<?x8192xf32>
        %6 = affine.apply affine_map<()[s0] -> (s0 * 256 - (s0 floordiv 32) * 8192)>()[%workgroup_id_x]
        %7 = flow.dispatch.tensor.load %2, offsets = [0, %6], sizes = [8192, %c256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x8192xf32>> -> tensor<8192x?xf32>
        %8 = affine.apply affine_map<()[s0] -> ((s0 floordiv 32) * 128)>()[%workgroup_id_x]
        %9 = affine.apply affine_map<()[s0] -> (s0 * 256 - (s0 floordiv 32) * 8192)>()[%workgroup_id_x]
        %10 = flow.dispatch.tensor.load %0, offsets = [%8, %9], sizes = [%c128, %c256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x8192xf32>> -> tensor<?x?xf32>
        %cast = tensor.cast %10 : tensor<?x?xf32> to tensor<128x256xf32>
        %11 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 256, 32]]>} ins(%cst : f32) outs(%cast : tensor<128x256xf32>) -> tensor<128x256xf32>
        %cast_0 = tensor.cast %7 : tensor<8192x?xf32> to tensor<8192x256xf32>
        %cast_1 = tensor.cast %5 : tensor<?x8192xf32> to tensor<128x8192xf32>
        %12 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 256, 32]]>} ins(%cast_1, %cast_0 : tensor<128x8192xf32>, tensor<8192x256xf32>) outs(%11 : tensor<128x256xf32>) -> tensor<128x256xf32>
        %13 = tensor.empty() : tensor<128x256xf32>
        %14 = tensor.empty() : tensor<128x256xf32>
        %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%14, %12 : tensor<128x256xf32>, tensor<128x256xf32>) outs(%13 : tensor<128x256xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 256, 32]]>} {
        ^bb0(%in: f32, %in_3: f32, %out: f32):
          %18 = arith.addf %in, %in_3 : f32
          linalg.yield %18 : f32
        } -> tensor<128x256xf32>
        %cast_2 = tensor.cast %15 : tensor<128x256xf32> to tensor<?x?xf32>
        %16 = affine.apply affine_map<()[s0] -> ((s0 floordiv 32) * 128)>()[%workgroup_id_x]
        %17 = affine.apply affine_map<()[s0] -> (s0 * 256 - (s0 floordiv 32) * 8192)>()[%workgroup_id_x]
        flow.dispatch.tensor.store %cast_2, %3, offsets = [%16, %17], sizes = [%c128, %c256], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<8192x8192xf32>>
        return
      }
    }
  }
}

// CHECK: __iree_matmul_tf32_tf32_f32_128_256_32_3_true_true

// -----

hal.executable private @matmul_no_tileconfig {
  hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}> {    
    hal.executable.export public @matmul_no_tileconfig ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>]>]>) attributes {translation_info = #iree_codegen.translation_info<LLVMGPUMatmulTensorCore pipeline_depth = 3>, workgroup_size = [128 : index, 2 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %c2048 = arith.constant 2048 : index
      %c1 = arith.constant 1 : index
      hal.return %c2048, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @matmul_no_tileconfig() {
        %c128 = arith.constant 128 : index
        %c256 = arith.constant 256 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x8192xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x8192xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<8192x8192xf32>>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %3 = affine.apply affine_map<()[s0] -> ((s0 floordiv 32) * 128)>()[%workgroup_id_x]
        %4 = flow.dispatch.tensor.load %0, offsets = [%3, 0], sizes = [%c128, 8192], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x8192xf32>> -> tensor<?x8192xf32>
        %5 = affine.apply affine_map<()[s0] -> (s0 * 256 - (s0 floordiv 32) * 8192)>()[%workgroup_id_x]
        %6 = flow.dispatch.tensor.load %1, offsets = [0, %5], sizes = [8192, %c256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x8192xf32>> -> tensor<8192x?xf32>
        %7 = affine.apply affine_map<()[s0] -> ((s0 floordiv 32) * 128)>()[%workgroup_id_x]
        %8 = affine.apply affine_map<()[s0] -> (s0 * 256 - (s0 floordiv 32) * 8192)>()[%workgroup_id_x]
        %9 = flow.dispatch.tensor.load %2, offsets = [%7, %8], sizes = [%c128, %c256], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<8192x8192xf32>> -> tensor<?x?xf32>
        %10 = linalg.matmul  ins(%4, %6 : tensor<?x8192xf32>, tensor<8192x?xf32>) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %11 = affine.apply affine_map<()[s0] -> ((s0 floordiv 32) * 128)>()[%workgroup_id_x]
        %12 = affine.apply affine_map<()[s0] -> (s0 * 256 - (s0 floordiv 32) * 8192)>()[%workgroup_id_x]
        flow.dispatch.tensor.store %10, %2, offsets = [%11, %12], sizes = [%c128, %c256], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<readwrite:tensor<8192x8192xf32>>
        return
      }
    }
  }
}

// CHECK: linalg.matmul

// -----

hal.executable private @matmul_no_pipeline {
  hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}> {
    hal.executable.export public @matmul_alone ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device):
      %c2048 = arith.constant 2048 : index
      %c1 = arith.constant 1 : index
      hal.return %c2048, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @matmul_alone() {
        %c128 = arith.constant 128 : index
        %c256 = arith.constant 256 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x8192xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x8192xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<8192x8192xf32>>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %3 = affine.apply affine_map<()[s0] -> ((s0 floordiv 32) * 128)>()[%workgroup_id_x]
        %4 = flow.dispatch.tensor.load %0, offsets = [%3, 0], sizes = [%c128, 8192], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x8192xf32>> -> tensor<?x8192xf32>
        %5 = affine.apply affine_map<()[s0] -> (s0 * 256 - (s0 floordiv 32) * 8192)>()[%workgroup_id_x]
        %6 = flow.dispatch.tensor.load %1, offsets = [0, %5], sizes = [8192, %c256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x8192xf32>> -> tensor<8192x?xf32>
        %7 = affine.apply affine_map<()[s0] -> ((s0 floordiv 32) * 128)>()[%workgroup_id_x]
        %8 = affine.apply affine_map<()[s0] -> (s0 * 256 - (s0 floordiv 32) * 8192)>()[%workgroup_id_x]
        %9 = flow.dispatch.tensor.load %2, offsets = [%7, %8], sizes = [%c128, %c256], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<8192x8192xf32>> -> tensor<?x?xf32>
        %10 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 256, 32]]>} ins(%4, %6 : tensor<?x8192xf32>, tensor<8192x?xf32>) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %11 = affine.apply affine_map<()[s0] -> ((s0 floordiv 32) * 128)>()[%workgroup_id_x]
        %12 = affine.apply affine_map<()[s0] -> (s0 * 256 - (s0 floordiv 32) * 8192)>()[%workgroup_id_x]
        flow.dispatch.tensor.store %10, %2, offsets = [%11, %12], sizes = [%c128, %c256], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<readwrite:tensor<8192x8192xf32>>
        return
      }
    }
  }
}

// CHECK: linalg.matmul