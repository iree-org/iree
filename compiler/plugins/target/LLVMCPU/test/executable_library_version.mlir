// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=generic --iree-hal-link-executables=false --iree-hal-dump-executable-intermediates-to=%t -o %t.vmfb
// RUN: cat %t/*plain*.linked.ll | FileCheck %s --check-prefix=PLAIN
// RUN: cat %t/*checked*.linked.ll | FileCheck %s --check-prefix=STATUS

// A library is v0.6 unless an export returns a status, which needs v0.7; its
// query answers any runtime at least that new.
// PLAIN: @iree_hal_executable_library_query_v0_header = {{.*}}{ i32 6,
// PLAIN: define {{.*}} @iree_hal_executable_library_query(i32 %0,
// PLAIN: icmp uge i32 %0, 6
// STATUS: @iree_hal_executable_library_query_v0_header = {{.*}}{ i32 7,
// STATUS: define {{.*}} @iree_hal_executable_library_query(i32 %0,
// STATUS: icmp uge i32 %0, 7

func.func @plain(%lhs: tensor<4x3xf32>, %rhs: tensor<4x3xf32>) -> tensor<4x3xf32> {
  %result = arith.addf %lhs, %rhs : tensor<4x3xf32>
  return %result : tensor<4x3xf32>
}

func.func @checked(%input: tensor<4x3xf32>) -> tensor<4x3xf32> {
  %zero = arith.constant 0.0 : f32
  %ok = arith.constant 0 : i32
  %failed = arith.constant 9 : i32
  %empty = tensor.empty() : tensor<4x3xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%input : tensor<4x3xf32>) outs(%empty : tensor<4x3xf32>) {
  ^bb0(%in: f32, %out: f32):
    %nonnegative = arith.cmpf oge, %in, %zero : f32
    %code = arith.select %nonnegative, %ok, %failed : i32
    util.status.check_ok %code, "negative input"
    linalg.yield %in : f32
  } -> tensor<4x3xf32>
  return %result : tensor<4x3xf32>
}
