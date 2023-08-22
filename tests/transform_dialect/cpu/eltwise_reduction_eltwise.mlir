!in_tensor_t = tensor<32x256xf32>
!out_tensor_t = tensor<32xf32>

func.func @reduce(%arg : !in_tensor_t) -> (!out_tensor_t) {
  %cst = arith.constant -0.000000e+00 : f32

  %0 = tensor.empty() : !out_tensor_t
  %1 = linalg.fill ins(%cst : f32) outs(%0 : !out_tensor_t) ->  !out_tensor_t
  %2 = tensor.empty() : !in_tensor_t
  %3 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg : !in_tensor_t) outs(%2 : !in_tensor_t) {
    ^bb0(%arg3: f32, %arg4: f32):
      %4 = arith.addf %arg3, %arg3 : f32
      %5 = arith.addf %4, %4 : f32
      linalg.yield %5 : f32
    } -> !in_tensor_t

  %6 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%3 : !in_tensor_t) outs(%1 : !out_tensor_t) {
      ^bb0(%arg3: f32, %arg4: f32):
        %4 = arith.addf %arg3, %arg4 : f32
        linalg.yield %4 : f32
      } -> !out_tensor_t

  %7 = tensor.empty() : !out_tensor_t
  %8 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%6 : !out_tensor_t) outs(%7 : !out_tensor_t) {
    ^bb0(%arg3: f32, %arg4: f32):
      %4 = math.sqrt %arg3 : f32
      linalg.yield %4 : f32
    } -> !out_tensor_t


  return %8 : !out_tensor_t
}


// RUN: iree-opt %s --iree-hal-target-backends=llvm-cpu \
// RUN:     --iree-abi-transformation-pipeline \
// RUN:     --iree-flow-transformation-pipeline \
// RUN:     --iree-stream-transformation-pipeline \
// RUN:     --iree-hal-configuration-pipeline | \
// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-codegen-materialize-user-configs, iree-llvmcpu-lower-executable-target)))' \
// RUN:     --iree-codegen-llvmcpu-enable-transform-dialect-jit | \
// RUN: FileCheck %s

// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu  \
// RUN:     --iree-codegen-llvmcpu-enable-transform-dialect-jit | \
// RUN: iree-run-module --module=- --function=reduce --device=local-task --input="32x256xf32=1" |\
// RUN: FileCheck %s --check-prefix=EXEC

//      CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//      CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//      CHECK-DAG: %[[workgroup_id_x:.*]] = hal.interface.workgroup.id[0] : index
//          CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} -> (vector<8xf32>) {
//          CHECK:   arith.addf %{{.*}} : vector<8x16xf32>
// CHECK-COUNT-16:   vector.extract %{{.*}} : vector<8xf32> from vector<16x8xf32>{{[[:space:]].*}}arith.addf %{{.*}} : vector<8xf32>
//          CHECK:   scf.yield %{{.*}} : vector<8xf32>
//          CHECK: }
//          CHECK: math.sqrt %{{.*}} : vector<8xf32>
//          CHECK: vector.store %{{.*}} : memref<8xf32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>, vector<8xf32>

//      EXEC: result[0]: hal.buffer_view
// EXEC-NEXT: 32xf32=32 32 32 32 32 32 32 32
