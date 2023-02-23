// RUN: iree-opt %s --pass-pipeline="builtin.module(hal.executable(iree-transform-dialect-interpreter,transform-dialect-drop-schedule))" | FileCheck %s

#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}>
#map = affine_map<()[s0] -> (s0 * 8)>
#map1 = affine_map<(d0) -> (d0)>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>
#translation = #iree_codegen.translation_info<TransformDialectCodegen>
hal.executable private @distribute {
  hal.executable.variant public @cuda_nvptx_fb, target = #executable_target_cuda_nvptx_fb {
    hal.executable.export public @distribute ordinal(0) layout(#pipeline_layout) attributes {translation_info = #translation} {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %c1 = arith.constant 1 : index
      hal.return %arg1, %c1, %c1 : index, index, index
    }
    builtin.module {
//   CHECK-DAG:  #[[$MAP:.+]] = affine_map<()[s0, s1] -> ((s1 * 2 + s0 floordiv 32) mod 4)>
//   CHECK-DAG:  #[[$MAP1:.+]] = affine_map<()[s0, s1] -> ((s1 * 2 + s0 floordiv 32) floordiv 4)>

// CHECK-LABEL: func.func @distribute
      func.func @distribute() {
        %cst_0 = arith.constant dense<0.000000e+00> : vector<1xf16>
        %c60 = arith.constant 60 : index
        %c8 = arith.constant 8 : index
        %c4 = arith.constant 4 : index
        %c2 = arith.constant 2 : index
        %c0 = arith.constant 0 : index
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<1024x1024xf16>
// CHECK: %[[TX:.+]] = gpu.thread_id  x
// CHECK: %[[TY:.+]] = gpu.thread_id  y
// CHECK: %[[COND:.*]] = arith.cmpi ult
// CHECK: scf.if %[[COND]] {
// CHECK:   vector.transfer_write %{{.*}}, %{{.*}}[%[[TY]], %[[TX]]] {in_bounds = [true]} : vector<1xf16>, memref<1024x1024xf16>
        scf.forall (%arg0, %arg1) in (%c4, %c60) {
          vector.transfer_write %cst_0, %1[%arg0, %arg1]
          {in_bounds = [true]} : vector<1xf16>, memref<1024x1024xf16>
        } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
// CHECK: %[[WX:.+]] = affine.apply #[[$MAP]]()[%[[TX]], %[[TY]]]
// CHECK: %[[WY:.+]] = affine.apply #[[$MAP1]]()[%[[TX]], %[[TY]]]
// CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[WY]], %[[WX]]] {in_bounds = [true]} : vector<1xf16>, memref<1024x1024xf16>
        scf.forall (%arg0, %arg1) in (%c2, %c4) {
          vector.transfer_write %cst_0, %1[%arg0, %arg1]
          {in_bounds = [true]} : vector<1xf16>, memref<1024x1024xf16>
        } {mapping = [#gpu.warp<y>, #gpu.warp<x>]}        
        return
      }
      module {
        transform.structured.canonicalized_sequence failures(propagate) {
        ^bb0(%arg0: !pdl.operation):
        %17 = transform.structured.match ops{["func.func"]} in %arg0 : (!pdl.operation) -> !pdl.operation
        %18 = transform.iree.map_nested_forall_to_gpu_threads %17 {workgroup_size = [64, 4, 1]}
      }
    }
  }
}
}
