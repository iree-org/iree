// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-spirv-tile-and-distribute)))))' %s | FileCheck %s


#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>
hal.executable private @repro {
  hal.executable.variant @vulkan target(<"vulkan-spirv", "vulkan-spirv-fb">) {
    hal.executable.export public @online_attention_buffers layout(#layout)
    builtin.module {
      func.func @online_attention_buffers(%q: memref<1x16x16xf32>, %k: memref<1x16x16xf32>, %v: memref<1x16x16xf32>,
                      %acc: memref<1x16x16xf32>, %max: memref<1x16xf32>, %sum: memref<1x16xf32>)
          attributes {translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<BaseDistribute> workgroup_size = [32, 1, 1]>} {
        %scale = arith.constant 0.25 : f32
        iree_linalg_ext.online_attention {indexing_maps = [
          affine_map<(h,m,d,k,n)->(h,m,d)>,
          affine_map<(h,m,d,k,n)->(h,k,d)>,
          affine_map<(h,m,d,k,n)->(h,k,n)>,
          affine_map<(h,m,d,k,n)->()>,
          affine_map<(h,m,d,k,n)->(h,m,n)>,
          affine_map<(h,m,d,k,n)->(h,m)>,
          affine_map<(h,m,d,k,n)->(h,m)>],
          lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 0, 0, 16], [1, 1, 0, 0, 1], [0, 0, 16, 16, 0]]>}
          ins(%q, %k, %v, %scale : memref<1x16x16xf32>, memref<1x16x16xf32>, memref<1x16x16xf32>, f32)
          outs(%acc, %max, %sum : memref<1x16x16xf32>, memref<1x16xf32>, memref<1x16xf32>) {
        ^bb0(%score: f32):
          iree_linalg_ext.yield %score : f32
        }
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @online_attention_buffers(
// CHECK: gpu.thread_id x
// CHECK: gpu.block_dim x
// CHECK: scf.for
// CHECK: iree_linalg_ext.online_attention
// CHECK-SAME: outs(
// CHECK: iree_linalg_ext.yield
