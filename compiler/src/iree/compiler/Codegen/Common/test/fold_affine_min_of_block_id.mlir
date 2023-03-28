// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-fold-affinemin-in-distributed-loops, canonicalize)))))' --split-input-file %s | FileCheck %s

hal.executable public @generic_static {
  hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}> {
    hal.executable.export public @generic_static ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) attributes {translation_info = #iree_codegen.translation_info<LLVMGPUTransposeSharedMem>, workgroup_size = [8 : index, 32 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %c128 = arith.constant 128 : index
      %c1 = arith.constant 1 : index
      hal.return %c128, %c128, %c1 : index, index, index
    }
    builtin.module {
      func.func @generic_static() {
// CHECK-LABEL: func.func @generic_static
//   CHECK-NOT:   affine.min
//   CHECK-NOT:   affine.min
//       CHECK: linalg.generic
//       CHECK:} -> tensor<32x32xf32>
//       CHECK: flow.dispatch.tensor.store {{.*}} sizes = [32, 32], strides = [1, 1] : tensor<32x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<4096x4096xf32>>
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<4096x4096xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<4096x4096xf32>>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %2 = affine.min affine_map<()[s0] -> (32, s0 * -32 + 4096)>()[%workgroup_id_y]
        %3 = affine.min affine_map<()[s0] -> (32, s0 * -32 + 4096)>()[%workgroup_id_x]
        %4 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
        %5 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
        %6 = flow.dispatch.tensor.load %0, offsets = [%4, %5], sizes = [%3, %2], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x4096xf32>> -> tensor<?x?xf32>
        %7 = tensor.empty(%2, %3) : tensor<?x?xf32>
        %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<?x?xf32>) outs(%7 : tensor<?x?xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32]]>} {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<?x?xf32>
        %9 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
        %10 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
        flow.dispatch.tensor.store %8, %1, offsets = [%9, %10], sizes = [%2, %3], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<4096x4096xf32>>
        return
      }
    }
  }
}
