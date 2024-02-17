// RUN: iree-opt %s --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-transform-dialect-interpreter)))' \
// RUN:   --iree-codegen-transform-dialect-library=%p/attention_mfma_transform_spec.mlir | \
// RUN: FileCheck --check-prefix=CHECK %s

hal.executable private @attention {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @attention_dispatch_0_attention_16x16384x128xf16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @attention_dispatch_0_attention_16x16384x128xf16() {
        // CHECK-NOT: vector.contract
        // CHECK-NOT: iree_vector_ext.to_simd
        // CHECK-NOT: iree_vector_ext.to_simt
        // CHECK: scf.for {{.*}} = %c0 to %c16384 step %c64 {{.*}} -> (vector<2xf32>, vector<2xf32>, vector<8x2x4xf32>)
        // CHECK-COUNT-128: amdgu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32}
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x16384x128xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x16384x128xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x16384x128xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x16384x128xf16>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [16, 16384, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x16384x128xf16>> -> tensor<16x16384x128xf16>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [16, 16384, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x16384x128xf16>> -> tensor<16x16384x128xf16>
        %6 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [16, 16384, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x16384x128xf16>> -> tensor<16x16384x128xf16>
        %7 = tensor.empty() : tensor<16x16384x128xf16>
        %8 = iree_linalg_ext.attention ins(%4, %5, %6 : tensor<16x16384x128xf16>, tensor<16x16384x128xf16>, tensor<16x16384x128xf16>) outs(%7 : tensor<16x16384x128xf16>) -> tensor<16x16384x128xf16>
        flow.dispatch.tensor.store %8, %3, offsets = [0, 0, 0], sizes = [16, 16384, 128], strides = [1, 1, 1] : tensor<16x16384x128xf16> -> !flow.dispatch.tensor<writeonly:tensor<16x16384x128xf16>>
        return
      }
    }
  }
}
