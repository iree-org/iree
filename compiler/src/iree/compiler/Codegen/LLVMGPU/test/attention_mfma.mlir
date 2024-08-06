// RUN: iree-opt %s --pass-pipeline='builtin.module(iree-transform-dialect-interpreter)' \
// RUN:   --iree-gpu-test-target=gfx908 --iree-codegen-transform-dialect-library=%p/attention_mfma_transform_spec.mlir | \
// RUN: FileCheck --check-prefix=CHECK %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
func.func @attention_dispatch_0_attention_16x16384x128xf16() {
  %c0 = arith.constant 0 : index
  %scale = arith.constant 0.08838834764 : f16
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x16384x128xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x16384x128xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x16384x128xf16>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(3) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x16384x128xf16>>
  %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [16, 16384, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x16384x128xf16>> -> tensor<16x16384x128xf16>
  %5 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [16, 16384, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x16384x128xf16>> -> tensor<16x16384x128xf16>
  %6 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [16, 16384, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x16384x128xf16>> -> tensor<16x16384x128xf16>
  %7 = tensor.empty() : tensor<16x16384x128xf16>
  %8 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                    affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                    affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                    affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
                    ins(%4, %5, %6, %scale : tensor<16x16384x128xf16>, tensor<16x16384x128xf16>, tensor<16x16384x128xf16>, f16) outs(%7 : tensor<16x16384x128xf16>) -> tensor<16x16384x128xf16>
  flow.dispatch.tensor.store %8, %3, offsets = [0, 0, 0], sizes = [16, 16384, 128], strides = [1, 1, 1] : tensor<16x16384x128xf16> -> !flow.dispatch.tensor<writeonly:tensor<16x16384x128xf16>>
  return
}
    // CHECK-NOT: vector.contract
    // CHECK-NOT: iree_vector_ext.to_simd
    // CHECK-NOT: iree_vector_ext.to_simt
    // CHECK-COUNT-8: vector.load {{.*}} : memref<16x16384x128xf16, #hal.descriptor_type<storage_buffer>>, vector<8xf16>
    // CHECK: scf.for {{.*}} = %c0 to %c16384 step %c64 {{.*}} -> (vector<2xf32>, vector<2xf32>, vector<8x2x4xf32>)
    // CHECK-COUNT-16: vector.load {{.*}} : memref<64x128xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    // CHECK-COUNT-128: amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32}
