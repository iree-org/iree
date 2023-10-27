// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-gpu-distribute, cse)))))" %s | FileCheck %s

hal.executable private @add_tensor  {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>) {
  hal.executable.export public @add_tensor ordinal(0)
  layout(#hal.pipeline.layout<push_constants = 0,
         sets = [<0, bindings = [<0, storage_buffer>, <1, storage_buffer>, <2, storage_buffer>]>]>)
         attributes {translation_info = #iree_codegen.translation_info<LLVMGPUVectorize>,
                     workgroup_size = [64 : index, 1 : index, 1 : index]} {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @add_tensor() {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<233x1024xf32>
      memref.assume_alignment %0, 64 : memref<233x1024xf32>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<233x1024xf32>
      memref.assume_alignment %1, 64 : memref<233x1024xf32>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<233x1024xf32>
      memref.assume_alignment %2, 64 : memref<233x1024xf32>
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %workgroup_id_y = hal.interface.workgroup.id[1] : index
      %3 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%workgroup_id_x]
      %4 = memref.subview %2[%workgroup_id_y, %3] [1, 256] [1, 1] : memref<233x1024xf32> to memref<1x256xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
      %5 = memref.subview %0[%workgroup_id_y, %3] [1, 256] [1, 1] : memref<233x1024xf32> to memref<1x256xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
      %6 = memref.subview %1[%workgroup_id_y, %3] [1, 256] [1, 1] : memref<233x1024xf32> to memref<1x256xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
      scf.forall (%arg0) in (%c64) shared_outs() -> () {
        %7 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg0)
        %8 = memref.subview %4[0, %7] [1, 4] [1, 1] : memref<1x256xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>> to memref<1x4xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
        %9 = vector.transfer_read %5[%c0, %7], %cst {in_bounds = [true]} : memref<1x256xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>, vector<4xf32>
        %10 = vector.transfer_read %6[%c0, %7], %cst {in_bounds = [true]} : memref<1x256xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>, vector<4xf32>
        %11 = arith.addf %9, %10 : vector<4xf32>
        vector.transfer_write %11, %8[%c0, %c0] {in_bounds = [true]} : vector<4xf32>, memref<1x4xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
      } {mapping = [#gpu.thread<x>]}
      return
    }
  }
}
}

//         CHECK: #[[$MAP:.*]] = affine_map<(d0) -> (d0 * 4)>
//   CHECK-LABEL: func.func @add_tensor
//         CHECK:   %[[C0:.*]] = arith.constant 0 : index
//         CHECK:   %[[TX:.*]] = gpu.thread_id  x
//         CHECK:   %[[OFF:.*]] = affine.apply #[[$MAP]](%[[TX]])
//         CHECK:   %[[S:.*]] = memref.subview %{{.*}}[0, %[[OFF]]] [1, 4] [1, 1] : memref<1x256xf32, #{{.*}}> to memref<1x4xf32, #{{.*}}>
//         CHECK:   %[[A:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[OFF]]], %{{.*}} {in_bounds = [true]} : memref<1x256xf32, #{{.*}}>, vector<4xf32>
//         CHECK:   %[[B:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[OFF]]], %{{.*}} {in_bounds = [true]} : memref<1x256xf32, #{{.*}}>, vector<4xf32>
//         CHECK:   %[[C:.*]] = arith.addf %[[A]], %[[B]] : vector<4xf32>
//         CHECK:   vector.transfer_write %[[C]], %[[S]][%[[C0]], %[[C0]]] {in_bounds = [true]} : vector<4xf32>, memref<1x4xf32, #{{.*}}>
