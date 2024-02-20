// RUN: iree-opt --split-input-file --iree-codegen-llvmgpu-use-vector-distribution \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-select-lowering-strategy, iree-llvmgpu-lower-executable-target)))" %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @matmul_256x256x256 {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb", {
      target_arch = "gfx940",
      mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>,
                        #iree_gpu.mfma_layout<F16_32x32x8_F32>]
  }>) {
  hal.executable.export @matmul_256x256x256 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @matmul_256x256x256() {
      %cst = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xf16>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xf16>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<256x256xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xf16>> -> tensor<256x256xf16>
      %5 = tensor.empty() : tensor<256x256xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256x256xf32>) -> tensor<256x256xf32>
      %7 = linalg.matmul ins(%3, %4 : tensor<256x256xf16>, tensor<256x256xf16>) outs(%6 : tensor<256x256xf32>) -> tensor<256x256xf32>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : tensor<256x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<256x256xf32>>
      return
    }
  }
}
}

// Basic pipeline test to make sure it generates the instructions we expect.

//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUVectorDistribute>
// CHECK-LABEL: hal.executable.export public @matmul_256x256x256
//  CHECK-SAME:    mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 2, subgroup_m_tile_count = 2, subgroup_n_tile_count = 4, subgroup_k_tile_count = 2>
//  CHECK-SAME:    subgroup_size = 64
//  CHECK-SAME:    translation_info = #[[$TRANSLATION]]
//  CHECK-SAME:    workgroup_size = [128 : index, 2 : index, 1 : index]

//    CHECK-LABEL: func.func @matmul_256x256x256
//          CHECK:   scf.for {{.*}} = %c0 to %c256 step %c32 iter_args({{.*}}) -> (vector<2x4x1x1x1x4xf32>)
// Each subgroup handles 2 * 4 tiles, and for each tile we accumulate 2 times
// along the K dimension. So in total 16 mfma ops.
// CHECK-COUNT-16:     amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
//          CHECK:     scf.yield %{{.+}} : vector<2x4x1x1x1x4xf32>
//  CHECK-COUNT-8:   vector.transfer_write {{.+}} {in_bounds = [true, true]} : vector<4x1xf32>, memref<256x256xf32, #hal.descriptor_type<storage_buffer>>
