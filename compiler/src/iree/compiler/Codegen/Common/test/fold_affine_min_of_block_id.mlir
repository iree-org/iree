// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-fold-affinemin-in-distributed-loops, canonicalize)))))' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable public @generic_static {
  hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export public @generic_static ordinal(0) layout(#pipeline_layout) {
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
//       CHECK: iree_tensor_ext.dispatch.tensor.store {{.*}} sizes = [32, 32], strides = [1, 1] : tensor<32x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x4096xf32>>
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x4096xf32>>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %2 = affine.min affine_map<()[s0] -> (32, s0 * -32 + 4096)>()[%workgroup_id_y]
        %3 = affine.min affine_map<()[s0] -> (32, s0 * -32 + 4096)>()[%workgroup_id_x]
        %4 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
        %5 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
        %6 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [%4, %5], sizes = [%3, %2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf32>> -> tensor<?x?xf32>
        %7 = tensor.empty(%2, %3) : tensor<?x?xf32>
        %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<?x?xf32>) outs(%7 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<?x?xf32>
        %9 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
        %10 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
        iree_tensor_ext.dispatch.tensor.store %8, %1, offsets = [%9, %10], sizes = [%2, %3], strides = [1, 1] : tensor<?x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x4096xf32>>
        return
      }
    }
  }
}
