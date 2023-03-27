// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))" --iree-codegen-llvmgpu-enable-transform-dialect-jit --iree-codegen-llvmgpu-enable-implicit-gemm %s | FileCheck %s

hal.executable @conv2d_nchw_fchw {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}> {
  hal.executable.export public @conv2d_nchw_fchw ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @conv2d_nchw_fchw() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x16x130x130xf16>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x16x3x3xf16>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x32x128x128xf16>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 16, 130, 130], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x16x130x130xf16>> -> tensor<2x16x130x130xf16>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [32, 16, 3, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x16x3x3xf16>> -> tensor<32x16x3x3xf16>
      %5 = tensor.empty() : tensor<2x32x128x128xf16>
      %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2x32x128x128xf16>) -> tensor<2x32x128x128xf16>
      %7 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%3, %4 : tensor<2x16x130x130xf16>, tensor<32x16x3x3xf16>) outs(%6 : tensor<2x32x128x128xf16>) -> tensor<2x32x128x128xf16>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 32, 128, 128], strides = [1, 1, 1, 1] : tensor<2x32x128x128xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x32x128x128xf16>>
      return
    }
  }
}
}

// CHECK-LABEL: func.func @conv2d_nchw_fchw
// CHECK:     %[[C0:.+]] = arith.constant 0 : index
// CHECK:     %[[C16:.+]] = arith.constant 16 : index
// CHECK:     %[[C144:.+]] = arith.constant 144 : index

// CHECK:     %[[ALLOC:.+]] = memref.alloc() {{.*}} : memref<1x16x32xf16, #gpu.address_space<workgroup>>
// CHECK:     %[[ALLOC11:.+]] = memref.alloc() {{.*}} : memref<32x16xf16, #gpu.address_space<workgroup>>
// CHECK:     %[[INPUT:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%[[C0]]) flags(ReadOnly) : memref<2x16x130x130xf16>
// CHECK:     %[[OUTPUT:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%[[C0]]) : memref<2x32x16384xf16>
// CHECK:     %[[FILTER:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%[[C0]]) flags(ReadOnly) : memref<32x144xf16>
// CHECK:     %[[WGIDX:.+]] = hal.interface.workgroup.id[0] : index
// CHECK:     %[[WGIDY:.+]] = hal.interface.workgroup.id[1] : index
// CHECK:     %[[WGIDZ:.+]] = hal.interface.workgroup.id[2] : index
// CHECK:     %[[SUBVIEW12:.+]] = memref.subview %[[OUTPUT]][%[[WGIDX]], %{{.*}}, %{{.*}}] [1, 32, 32] [1, 1, 1] : memref<2x32x16384xf16> to memref<1x32x32xf16, strided<[524288, 16384, 1], offset: ?>>
// CHECK:     gpu.barrier
// CHECK:     scf.for %{{.*}} = %[[C0]] to %[[C144]] step %[[C16]] {

// Img2col
// CHECK:       %[[COLVEC:.+]] = vector.gather {{.*}} : memref<2x16x130x130xf16>, vector<16xindex>, vector<16xi1>, vector<16xf16> into vector<16xf16>
// CHECK:       vector.transfer_write %[[COLVEC]], {{.*}} : vector<16xf16>, memref<16xf16, strided<[32], offset: ?>, #gpu.address_space<workgroup>>
// CHECK:       gpu.barrier

// Shared memory copy
// CHECK:       %{{.*}} = vector.transfer_read {{.*}} : memref<1x8xf16, strided<[144, 1], offset: ?>>, vector<8xf16>
// CHECK:       vector.transfer_write {{.*}} : vector<8xf16>, memref<1x8xf16, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>>
// CHECK:       %{{.*}} = vector.transfer_read {{.*}} : memref<1x8xf16, strided<[144, 1], offset: ?>>, vector<8xf16>
// CHECK:       vector.transfer_write {{.*}} : vector<8xf16>, memref<1x8xf16, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>>
// CHECK:       gpu.barrier

// Matmul
// CHECK:       scf.forall ({{.*}}) in (1) {
// CHECK-COUNT-2: %{{.*}} = gpu.subgroup_mma_load_matrix %[[ALLOC11]]{{.*}} : memref<32x16xf16, #gpu.address_space<workgroup>> -> !gpu.mma_matrix<16x16xf16, "AOp">
// CHECK-COUNT-2: %{{.*}} = gpu.subgroup_mma_load_matrix %[[ALLOC]]{{.*}} : memref<1x16x32xf16, #gpu.address_space<workgroup>> -> !gpu.mma_matrix<16x16xf16, "BOp">
// CHECK-COUNT-4: %{{.*}} = gpu.subgroup_mma_load_matrix %[[SUBVIEW12]]{{.*}} : memref<1x32x32xf16, strided<[524288, 16384, 1], offset: ?>> -> !gpu.mma_matrix<16x16xf16, "COp">
// CHECK-COUNT-4: %{{.*}} = gpu.subgroup_mma_compute {{.*}} : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
// CHECK-COUNT-4: gpu.subgroup_mma_store_matrix {{.*}} %[[SUBVIEW12]]{{.*}} : !gpu.mma_matrix<16x16xf16, "COp">, memref<1x32x32xf16, strided<[524288, 16384, 1], offset: ?>>
// CHECK:       } {mapping = [#gpu.warp<x>]}
// CHECK:     }
// CHECK:     return

// -----

hal.executable @conv2d_nhwc_hwcf {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}> {
  hal.executable.export public @conv2d_nhwc_hwcf ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @conv2d_nhwc_hwcf() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x130x130x16xf16>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x16x32xf16>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x128x128x32xf16>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 130, 130, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x130x130x16xf16>> -> tensor<2x130x130x16xf16>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 16, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x16x32xf16>> -> tensor<3x3x16x32xf16>
      %5 = tensor.empty() : tensor<2x128x128x32xf16>
      %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2x128x128x32xf16>) -> tensor<2x128x128x32xf16>
      %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%3, %4 : tensor<2x130x130x16xf16>, tensor<3x3x16x32xf16>) outs(%6 : tensor<2x128x128x32xf16>) -> tensor<2x128x128x32xf16>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 128, 128, 32], strides = [1, 1, 1, 1] : tensor<2x128x128x32xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x128x128x32xf16>>
      return
    }
  }
}
}

// CHECK-LABEL: func.func @conv2d_nhwc_hwcf
// CHECK:     %[[C0:.+]] = arith.constant 0 : index
// CHECK:     %[[C16:.+]] = arith.constant 16 : index
// CHECK:     %[[C144:.+]] = arith.constant 144 : index

// CHECK:     %[[ALLOC:.+]] = memref.alloc() {{.*}} : memref<1x128x16xf16, #gpu.address_space<workgroup>>
// CHECK:     %[[ALLOC14:.+]] = memref.alloc() {{.*}} : memref<16x32xf16, #gpu.address_space<workgroup>>
// CHECK:     %[[INPUT:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%[[C0]]) flags(ReadOnly) : memref<2x130x130x16xf16>
// CHECK:     %[[OUTPUT:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%[[C0]]) : memref<2x16384x32xf16>
// CHECK:     %[[FILTER:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%[[C0]]) flags(ReadOnly) : memref<144x32xf16>
// CHECK:     %[[WGIDX:.+]] = hal.interface.workgroup.id[0] : index
// CHECK:     %[[WGIDY:.+]] = hal.interface.workgroup.id[1] : index
// CHECK:     %[[WGIDZ:.+]] = hal.interface.workgroup.id[2] : index
// CHECK:     %[[SUBVIEW15:.+]] = memref.subview %[[OUTPUT]][%[[WGIDX]], %{{.*}}, %{{.*}}] [1, 128, 32] [1, 1, 1] : memref<2x16384x32xf16> to memref<1x128x32xf16, strided<[524288, 32, 1], offset: ?>>
// CHECK:     gpu.barrier
// CHECK:     scf.for %{{.*}} = %[[C0]] to %[[C144]] step %[[C16]] {

// Img2col
// CHECK:       %[[COLVEC:.+]] = vector.gather {{.*}} : memref<2x130x130x16xf16>, vector<2x16xindex>, vector<2x16xi1>, vector<2x16xf16> into vector<2x16xf16>
// CHECK:       vector.transfer_write %[[COLVEC]], {{.*}} : vector<2x16xf16>, memref<1x2x16xf16, strided<[2048, 16, 1], offset: ?>, #gpu.address_space<workgroup>>
// CHECK:       gpu.barrier

// Shared memory copy
// CHECK:       %{{.*}} = vector.transfer_read {{.*}} : memref<1x8xf16, strided<[32, 1], offset: ?>>, vector<8xf16>
// CHECK:       vector.transfer_write {{.*}} : vector<8xf16>, memref<1x8xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>
// CHECK:       gpu.barrier

// Matmul
// CHECK:       scf.forall ({{.*}}) in (2) {
// CHECK:         %[[SUBVIEW21:.+]] = memref.subview %[[SUBVIEW15]]{{.*}} : memref<1x128x32xf16, strided<[524288, 32, 1], offset: ?>> to memref<1x64x32xf16, strided<[524288, 32, 1], offset: ?>>
// CHECK-COUNT-4: %{{.*}} = gpu.subgroup_mma_load_matrix %[[ALLOC]]{{.*}} : memref<1x128x16xf16, #gpu.address_space<workgroup>> -> !gpu.mma_matrix<16x16xf16, "AOp">
// CHECK-COUNT-2: %{{.*}} = gpu.subgroup_mma_load_matrix %[[ALLOC14]]{{.*}} : memref<16x32xf16, #gpu.address_space<workgroup>> -> !gpu.mma_matrix<16x16xf16, "BOp">
// CHECK-COUNT-8: %{{.*}} = gpu.subgroup_mma_load_matrix %[[SUBVIEW15]]{{.*}} : memref<1x128x32xf16, strided<[524288, 32, 1], offset: ?>> -> !gpu.mma_matrix<16x16xf16, "COp">
// CHECK-COUNT-8: %{{.*}} = gpu.subgroup_mma_compute {{.*}} : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
// CHECK-COUNT-8: gpu.subgroup_mma_store_matrix {{.*}} %[[SUBVIEW21]]{{.*}} : !gpu.mma_matrix<16x16xf16, "COp">, memref<1x64x32xf16, strided<[524288, 32, 1], offset: ?>>
// CHECK:       } {mapping = [#gpu.warp<x>]}
// CHECK:     }
// CHECK:     return
