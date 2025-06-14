// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-codegen-lower-executable-using-transform-dialect)" | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#translation = #iree_codegen.translation_info<pipeline = TransformDialectCodegen, { config_test = "config_test" }>
module {
  func.func @distribute() attributes {hal.executable.target = #executable_target_cuda_nvptx_fb, translation_info = #translation} {
    %cst = arith.constant dense<0.000000e+00> : vector<1xf16>
    %c250 = arith.constant 250 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<2xf16>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %subview = memref.subview %0[%workgroup_id_x] [1] [1] : memref<2xf16> to memref<1xf16, strided<[1], offset: ?>>
    scf.forall (%arg0) in (%c250) {
      vector.transfer_write %cst, %subview[%arg0] {in_bounds = [true]} : vector<1xf16>, memref<1xf16, strided<[1], offset: ?>>
    } {mapping = [#gpu.thread<x>]}
    scf.forall (%arg0) in (%c8) {
      vector.transfer_write %cst, %subview[%arg0] {in_bounds = [true]} : vector<1xf16>, memref<1xf16, strided<[1], offset: ?>>
    } {mapping = [#gpu.warp<x>]}
    return
  }
  builtin.module attributes { transform.with_named_sequence } {
    transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
      %17 = transform.structured.match ops{["func.func"]} in %variant_op
        : (!transform.any_op) -> !transform.any_op
      transform.iree.map_nested_forall_to_gpu_threads %17
        workgroup_dims = [256, 1, 1] subgroup_size = 32 : (!transform.any_op) -> ()

      // Late canonicalizations to cleanup and pass the checks.
      // Needs to occur on the whole variant to perform cse on the workgroup_count region
      %func_op = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
      transform.apply_patterns to %func_op {
        transform.apply_patterns.canonicalization
      } : !transform.any_op
      transform.iree.apply_licm %func_op : !transform.any_op
      transform.apply_cse to %func_op : !transform.any_op
      transform.yield
    } // @__transform_main
  } // module
}

// CHECK-DAG: #[[DIV32:.*]] = affine_map<()[s0] -> (s0 floordiv 32)>
// CHECK-DAG: #[[TRANSLATION_INFO:.*]] = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 1, 1] subgroup_size = 32, {config_test = "config_test"}>
// CHECK: func.func @distribute()
// CHECK-SAME: translation_info = #[[TRANSLATION_INFO]]
// CHECK: %[[TX:.+]] = gpu.thread_id  x
// CHECK: %[[COND:.*]] = arith.cmpi ult
// CHECK: scf.if %[[COND]] {
// CHECK:   vector.transfer_write %{{.*}}, %{{.*}}[%[[TX]]] {in_bounds = [true]} : vector<1xf16>, memref<1xf16, strided<[1], offset: ?>>
// CHECK: %[[WX:.+]] = affine.apply #[[DIV32]]()[%[[TX]]]
// CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[WX]]] {in_bounds = [true]} : vector<1xf16>, memref<1xf16, strided<[1], offset: ?>>
