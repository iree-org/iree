// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-transform-dialect-interpreter{library-file-name=%p/transform_dialect_codegen_vector_warp_execute_on_lane_0_spec.mlir})" \
// RUN: --allow-unregistered-dialect | \
// RUN: FileCheck %s --check-prefix=WARP-EXECUTE

// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-transform-dialect-interpreter{library-file-name=%p/transform_dialect_codegen_vector_distribution_spec.mlir})" \
// RUN: --allow-unregistered-dialect | \
// RUN: FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#translation_info = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 32>
func.func @reduce_dispatch_0() attributes {translation_info = #translation_info} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<128xf32>
  %1 = gpu.thread_id  x
  %2 = arith.cmpi ult, %1, %c1 : index

  // WARP-EXECUTE-DAG: %[[C0:.*]] = arith.constant 0 : index
  // WARP-EXECUTE-DAG: %[[C32:.*]] = arith.constant 32 : index
  // WARP-EXECUTE: %[[TIDX:.*]] = gpu.thread_id  x
  // WARP-EXECUTE: %[[COND32:.*]] = arith.cmpi ult, %[[TIDX]], %[[C32]] : index
  // Single-warp guard filters out threads 32-63.
  // WARP-EXECUTE: scf.if %[[COND32]] {
  // WARP-EXECUTE:   gpu.warp_execute_on_lane_0(%[[TIDX]])[32] {
  // WARP-EXECUTE:     %[[V:.*]] = "some_def"() : () -> vector<128xf32>
  // WARP-EXECUTE:     vector.transfer_write %[[V]], %{{.*}} {in_bounds = [true]} : vector<128xf32>, memref<128xf32>

  // CHECK-DAG: #[[MAP:.*]] = affine_map<()[s0] -> (s0 * 4)>
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
  // CHECK: %[[TIDX:.*]] = gpu.thread_id  x
  // CHECK: %[[COND32:.*]] = arith.cmpi ult, %[[TIDX]], %[[C32]] : index
  // Single-warp guard filters out threads 32-63.
  // CHECK: scf.if %[[COND32]] {
  // CHECK:   %[[COND1:.*]] = arith.cmpi eq, %[[TIDX]], %[[C0]] : index
  // CHECK:   %[[ALLOC:.*]] = memref.alloc() : memref<128xf32, #gpu.address_space<workgroup>>
  // Single-thread guard runs on thread 0 only.
  // CHECK:   scf.if %[[COND1]] {
  // CHECK:     %[[V:.*]] = "some_def"() : () -> vector<128xf32>
  // CHECK:     vector.transfer_write %[[V]], %{{.*}} : vector<128xf32>, memref<128xf32, #gpu.address_space<workgroup>>
  // CHECK:   %[[IDX:.*]] = affine.apply #[[MAP]]()[%[[TIDX]]]
  // CHECK:   %[[LOADED:.*]] = vector.transfer_read %{{.*}}[%[[IDX]]], %{{.*}} {in_bounds = [true]} : memref<128xf32, #gpu.address_space<workgroup>>, vector<4xf32>
  // CHECK:   vector.transfer_write %[[LOADED]], %{{.*}} {in_bounds = [true]} : vector<4xf32>, memref<128xf32>
  scf.if %2 {
    %v = "some_def"() : () -> (vector<128xf32>)
    vector.transfer_write %v, %0[%c0] : vector<128xf32>, memref<128xf32>
  }
  return
}
