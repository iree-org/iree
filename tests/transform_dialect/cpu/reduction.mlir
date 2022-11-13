!in_tensor_t = tensor<8x64xi8>
!out_tensor_t = tensor<8xi32>

func.func @reduce() -> (!out_tensor_t) {
  %cst = arith.constant 0 : i32

  // Note: arith.constant is good for our purposes here but it may be useful to use
  // util.unfoldable_constant.
  %arg = arith.constant dense<1> : !in_tensor_t
  %0 = tensor.empty() : !out_tensor_t
  %1 = linalg.fill ins(%cst : i32) outs(%0 : !out_tensor_t) ->   !out_tensor_t
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%arg : !in_tensor_t) outs(%1 : !out_tensor_t) {
      ^bb0(%arg3: i8, %arg4: i32):
        %3 = arith.extsi %arg3: i8 to i32
        %4 = arith.addi %3, %arg4 : i32
        linalg.yield %4 : i32
      } -> !out_tensor_t
  return %2 : !out_tensor_t
}

/// Note: the current --iree-codegen-llvmcpu-enable-transform-dialect-jit only works for exactly this reduction atm.
// RUN: iree-opt %s --iree-hal-target-backends=llvm-cpu \
// RUN:     --iree-abi-transformation-pipeline \
// RUN:     --iree-flow-transformation-pipeline  \
// RUN:     --iree-stream-transformation-pipeline \
// RUN:     --iree-hal-configuration-pipeline | \
// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target)))' \
// RUN:     --iree-codegen-llvmcpu-enable-transform-dialect-jit | \
// RUN: FileCheck %s --check-prefix=CHECK

/// Note: the current --iree-codegen-llvmcpu-enable-transform-dialect-jit only works for exactly this reduction atm.
// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu \
// RUN:     --iree-codegen-llvmcpu-enable-transform-dialect-jit | \
// RUN: iree-run-module --entry_function=reduce --device=local-task --task_topology_group_count=0 |\
// RUN: FileCheck %s --check-prefix=EXEC

// CHECK: func.func @reduce_dispatch_0_generic_8x64() {
/// 8x64 is parallelized 8-way and 64 is split into 2 parallel x 32 reduction
// CHECK: vector.transfer_read{{.*}}{in_bounds = [true, true]} : memref<2x32xi8>, vector<2x32xi8>
// CHECK: arith.extsi{{.*}}: vector<2x32xi8> to vector<2x32xi32>
// CHECK: vector.multi_reduction <add>{{.*}}[1] : vector<2x32xi32> to vector<2xi32>
// CHECK: vector.multi_reduction <add>{{.*}}[0] : vector<2xi32> to i32

//      EXEC: result[0]: hal.buffer_view
// EXEC-NEXT: 8xi32=64 64 64 64 64 64 64 64
