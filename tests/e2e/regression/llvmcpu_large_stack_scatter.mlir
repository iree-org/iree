// Checks that oversized scatter scratch is promoted, preserves input values,
// and runs on both local CPU drivers.
// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=generic --iree-opt-level=O2 --iree-llvmcpu-promote-allocas-to-scratch-memory=true -o %t.vmfb
// RUN: iree-run-module --module=%t.vmfb --device=local-sync --function=test | FileCheck %s --check-prefix=RESULT
// RUN: iree-run-module --module=%t.vmfb --device=local-task --task_worker_local_memory=65536 --function=test | FileCheck %s --check-prefix=RESULT
// RUN: not iree-run-module --module=%t.vmfb --device=local-task --task_worker_local_memory=4096 --function=test 2>&1 | FileCheck %s --check-prefix=CAPACITY
// RUN: not iree-compile %s --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=generic --iree-opt-level=O2 -o %t.disabled.vmfb 2>&1 | FileCheck %s --check-prefix=DISABLED
// RESULT: i32=1
// CAPACITY: dispatch requires 65536b of local memory but only 4096b is available per-worker
// DISABLED: exceeded stack allocation limit of 32768 bytes for function. Got 65536 bytes

func.func private @scatter(%cache: tensor<2x16384xf32>, %indices: tensor<1xi32>, %updates: tensor<1xf32>) -> tensor<16384xf32> attributes {util.noinline} {
  %slice = tensor.extract_slice %cache[0, 0] [1, 16384] [1, 1] : tensor<2x16384xf32> to tensor<16384xf32>
  %result = iree_linalg_ext.scatter dimension_map = [0] unique_indices(false)
      ins(%updates, %indices : tensor<1xf32>, tensor<1xi32>)
      outs(%slice : tensor<16384xf32>) {
    ^bb0(%update: f32, %old: f32):
      iree_linalg_ext.yield %update : f32
  } -> tensor<16384xf32>
  return %result : tensor<16384xf32>
}

func.func @test() -> i1 {
  %cache = util.unfoldable_constant dense<3.0> : tensor<2x16384xf32>
  %indices = util.unfoldable_constant dense<7> : tensor<1xi32>
  %updates = util.unfoldable_constant dense<42.0> : tensor<1xf32>
  %result = call @scatter(%cache, %indices, %updates) : (tensor<2x16384xf32>, tensor<1xi32>, tensor<1xf32>) -> tensor<16384xf32>
  %empty = tensor.empty() : tensor<16384xf32>
  %expected = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      outs(%empty : tensor<16384xf32>) {
    ^bb0(%unused: f32):
      %i = linalg.index 0 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3.0 : f32
      %c42 = arith.constant 42.0 : f32
      %is_updated = arith.cmpi eq, %i, %c7 : index
      %value = arith.select %is_updated, %c42, %c3 : f32
      linalg.yield %value : f32
  } -> tensor<16384xf32>
  %result_equal = arith.cmpf oeq, %result, %expected : tensor<16384xf32>
  %true = arith.constant dense<true> : tensor<i1>
  %result_ok = linalg.reduce ins(%result_equal : tensor<16384xi1>)
      outs(%true : tensor<i1>) dimensions = [0]
      (%element: i1, %acc: i1) {
    %both = arith.andi %element, %acc : i1
    linalg.yield %both : i1
  }
  %original = arith.constant dense<3.0> : tensor<2x16384xf32>
  %input_equal = arith.cmpf oeq, %cache, %original : tensor<2x16384xf32>
  %input_ok = linalg.reduce ins(%input_equal : tensor<2x16384xi1>)
      outs(%true : tensor<i1>) dimensions = [0, 1]
      (%element: i1, %acc: i1) {
    %both = arith.andi %element, %acc : i1
    linalg.yield %both : i1
  }
  %result_scalar = tensor.extract %result_ok[] : tensor<i1>
  %input_scalar = tensor.extract %input_ok[] : tensor<i1>
  %ok = arith.andi %result_scalar, %input_scalar : i1
  return %ok : i1
}
