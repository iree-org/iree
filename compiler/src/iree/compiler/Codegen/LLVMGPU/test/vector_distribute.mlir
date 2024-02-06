// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-vector-distribute, canonicalize, cse)))))' -split-input-file %s | FileCheck %s

hal.executable private @matmul_dispatch_0 {
  // This pass queries the target device for the list of supported mma ops.
  hal.executable.variant public @rocm_hsaco_fb target(
      <"rocm", "rocm-hsaco-fb", {
      mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>,
                        #iree_gpu.mfma_layout<F16_32x32x8_F32>],
      target_arch = "gfx940",
      ukernels = "none"}>) {
    hal.executable.export public @matmul_256x256x256 ordinal(0)
      layout(#hal.pipeline.layout<push_constants = 0,
             sets = [<0, bindings = [
                     <0, storage_buffer, ReadOnly>,
                     <1, storage_buffer, ReadOnly>,
                     <2, storage_buffer>]>]>)
      attributes {subgroup_size = 64 : index,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute>,
      workgroup_size = [64 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_256x256x256(%lhs: memref<16x256xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>,
                                    %rhs: memref<256x16xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>,
                                    %out: memref<16x16xf32, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>) {
        %alloc = memref.alloc() : memref<32x16xf16, #gpu.address_space<workgroup>>
        %alloc_0 = memref.alloc() : memref<16x32xf16, #gpu.address_space<workgroup>>
        %cst = arith.constant 0.000000e+00 : f16 
        %cst_1 = arith.constant dense<0.000000e+00> : vector<16x16xf32>
        %c32 = arith.constant 32 : index 
        %c256 = arith.constant 256 : index 
        %c0 = arith.constant 0 : index 
        %5 = scf.for %arg0 = %c0 to %c256 step %c32 iter_args(%arg1 = %cst_1) -> (vector<16x16xf32>) {
          %6 = vector.transfer_read %lhs[%c0, %arg0], %cst {in_bounds = [true, true]} : memref<16x256xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, vector<16x32xf16>
          %7 = vector.transfer_read %rhs[%arg0, %c0], %cst {in_bounds = [true, true]} : memref<256x16xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, vector<32x16xf16>
          vector.transfer_write %6, %alloc_0[%c0, %c0] {in_bounds = [true, true]} : vector<16x32xf16>, memref<16x32xf16, #gpu.address_space<workgroup>>
          gpu.barrier
          vector.transfer_write %7, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<32x16xf16>, memref<32x16xf16, #gpu.address_space<workgroup>>
          gpu.barrier
          %8 = vector.transfer_read %alloc_0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<16x32xf16, #gpu.address_space<workgroup>>, vector<16x32xf16>
          %9 = vector.transfer_read %alloc[%c0, %c0], %cst {in_bounds = [true, true]} : memref<32x16xf16, #gpu.address_space<workgroup>>, vector<32x16xf16>
          %10 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %8, %9, %arg1 : vector<16x32xf16>, vector<32x16xf16> into vector<16x16xf32> 
          scf.yield %10 : vector<16x16xf32>
        }
        vector.transfer_write %5, %out[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
        memref.dealloc %alloc_0 : memref<16x32xf16, #gpu.address_space<workgroup>>
        memref.dealloc %alloc : memref<32x16xf16, #gpu.address_space<workgroup>>
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @matmul_256x256x256
//       CHECK:   %[[INIT:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x4xf32>
//       CHECK:   %[[RHS_ALLOC:.+]] = memref.alloc() : memref<32x16xf16, #gpu.address_space<workgroup>>
//       CHECK:   %[[LHS_ALLOC:.+]] = memref.alloc() : memref<16x32xf16, #gpu.address_space<workgroup>>
//       CHECK:   scf.for {{.*}} = %c0 to %c256 step %c32 iter_args({{.*}} = %[[INIT]]) -> (vector<1x1x4xf32>)
//       CHECK:     %[[RLOAD:.+]] = vector.load %{{.*}} : memref<16x256xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, vector<8xf16>
//       CHECK:     %[[LLOAD:.+]] = vector.load %{{.*}} : memref<256x16xf16, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, vector<8xf16>
//       CHECK:     vector.store %[[RLOAD]], %[[LHS_ALLOC]]{{.*}} : memref<16x32xf16, #gpu.address_space<workgroup>>, vector<8xf16>
//       CHECK:     vector.store %[[LLOAD]], %[[RHS_ALLOC]]{{.*}} : memref<32x16xf16, #gpu.address_space<workgroup>>, vector<8xf16>
//       CHECK:     gpu.barrier
// CHECK-COUNT-2:   vector.load %[[LHS_ALLOC]]{{.*}} : memref<16x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
// CHECK-COUNT-8:   vector.load %[[RHS_ALLOC]]{{.*}} : memref<32x16xf16, #gpu.address_space<workgroup>>, vector<1xf16>
// CHECK-COUNT-2:   amdgpu.mfma {{.*}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32> 
//       CHECK:     %[[BCAST:.+]] = vector.broadcast {{.*}} : vector<4xf32> to vector<1x1x4xf32>
//       CHECK:     scf.yield %[[BCAST]] : vector<1x1x4xf32>
// CHECK-COUNT-4: vector.store {{.*}} : memref<16x16xf32{{.*}}>, vector<1xf32>
