// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx940 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-lower-executable-target)))))" %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0], reduction = [0, 0, 4], thread = [8, 4]}>
hal.executable public @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @matmul_transpose_b ordinal(0) layout(#pipeline_layout)
    attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_transpose_b()
        attributes {translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64>} {
        %cst = arith.constant 0.000000e+00 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<10240x1280xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>> -> tensor<2048x1280xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [10240, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<10240x1280xf16>> -> tensor<10240x1280xf16>
        %5 = tensor.empty() : tensor<2048x10240xf32>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2048x10240xf32>) -> tensor<2048x10240xf32>
        %7 = linalg.matmul_transpose_b {lowering_config = #config}
          ins(%3, %4 : tensor<2048x1280xf16>, tensor<10240x1280xf16>)
          outs(%6 : tensor<2048x10240xf32>) -> tensor<2048x10240xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 10240], strides = [1, 1] : tensor<2048x10240xf32> -> !flow.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
        return
      }
    }
  }
}

// Note that current barrier placement logic is observedly poor. Some cleanup
// analysis should be able to simplify the below to just two barriers.

// CHECK-LABEL: func @matmul_transpose_b
//   CHECK-DAG:   %[[B0:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[B1:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-DAG:   %[[B2:.+]] = hal.interface.binding.subspan set(0) binding(2)
//   CHECK-DAG:   memref.alloc() : memref<64x4xf16, #gpu.address_space<workgroup>>
//   CHECK-DAG:   memref.alloc() : memref<64x4xf16, #gpu.address_space<workgroup>>
//       CHECK:   %[[LOOP:.+]] = scf.for %[[IV:.+]] = %c0 to %c1280 step %c4 {{.*}} -> (vector<8x4xf32>)
//       CHECK:     gpu.barrier
//       CHECK:     %[[LHS_RD:.+]] = vector.transfer_read %[[B0]]{{.*}} vector<2xf16>
//       CHECK:     vector.transfer_write %[[LHS_RD]], %[[LHS_ALLOC:[A-Za-z0-9]+]]
//       CHECK:     gpu.barrier
//       CHECK:     %[[LHS_MM:.+]] = vector.transfer_read %[[LHS_ALLOC]]{{.*}} vector<8x4xf16>
//       CHECK:     gpu.barrier
//       CHECK:     %[[RHS_RD:.+]] = vector.transfer_read %[[B1]]{{.*}} vector<2xf16>
//       CHECK:     vector.transfer_write %[[RHS_RD]], %[[RHS_ALLOC:[A-Za-z0-9]+]]
//       CHECK:     gpu.barrier
//       CHECK:     %[[RHS_MM:.+]] = vector.transfer_read %[[RHS_ALLOC]]{{.*}} vector<4x4xf16>
//       CHECK:     gpu.barrier
//       CHECK:     %[[MM:.+]] = vector.contract {{.*}} %[[LHS_MM]], %[[RHS_MM]]
//       CHECK:     scf.yield %[[MM]]
//       CHECK:   vector.transfer_write %[[LOOP]], %[[B2]]
