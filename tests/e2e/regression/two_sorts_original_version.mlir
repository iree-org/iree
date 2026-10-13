// RUN: iree-run-mlir --Xcompiler,iree-hal-target-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --input="3xi64=[3 1 2]" | FileCheck %s
// CHECK-LABEL: EXEC @two_sorts_original_version
// CHECK: 3xi64=1 2 0
// CHECK: 3xi64=100 102 101
// CHECK: 3xi64=3 1 2
#identity = affine_map<(d0) -> (d0)>
func.func @two_sorts_original_version(%keys: tensor<?xi64>) -> (tensor<?xi64>, tensor<?xi64>, tensor<?xi64>) {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %keys, %c0 : tensor<?xi64>
  %empty0 = tensor.empty(%n) : tensor<?xi64>
  %empty1 = tensor.empty(%n) : tensor<?xi64>
  %indices0 = linalg.generic {indexing_maps = [#identity], iterator_types = ["parallel"]} outs(%empty0 : tensor<?xi64>) {
  ^bb0(%out: i64):
    %i = linalg.index 0 : index
    %value = arith.index_cast %i : index to i64
    linalg.yield %value : i64
  } -> tensor<?xi64>
  %indices1 = linalg.generic {indexing_maps = [#identity], iterator_types = ["parallel"]} outs(%empty1 : tensor<?xi64>) {
  ^bb0(%out: i64):
    %offset = arith.constant 100 : i64
    %i = linalg.index 0 : index
    %value = arith.index_cast %i : index to i64
    %label = arith.addi %value, %offset : i64
    linalg.yield %label : i64
  } -> tensor<?xi64>
  %results:2 = flow.dispatch.region -> (tensor<?xi64>{%n}, tensor<?xi64>{%n}) {
    %ascending:2 = iree_linalg_ext.sort dimension(0) outs(%keys, %indices0 : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    %descending:2 = iree_linalg_ext.sort dimension(0) outs(%keys, %indices1 : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
      %cmp = arith.cmpi sge, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    flow.return %ascending#1, %descending#1 : tensor<?xi64>, tensor<?xi64>
  }
  return %results#0, %results#1, %keys : tensor<?xi64>, tensor<?xi64>, tensor<?xi64>
}
