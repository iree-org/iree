// RUN: iree-run-mlir --Xcompiler,iree-hal-target-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --input="4xi64=[4 1 3 2]" --input="4xi64=[7 -3 9 11]" | FileCheck %s --check-prefix=EARLIER

#identity = affine_map<(d0) -> (d0)>
// An actual data observation of the earlier version must survive the update.
#scalar = affine_map<(d0) -> ()>
func.func @two_sorts_keep_internal_version(%keys: tensor<4xi64>, %labels: tensor<4xi64>)
    -> (tensor<4xi64>, tensor<i64>, tensor<4xi64>, tensor<4xi64>) {
  %zero = arith.constant 0 : i64
  %sum_empty = tensor.empty() : tensor<i64>
  %sum_init = linalg.fill ins(%zero : i64) outs(%sum_empty : tensor<i64>) -> tensor<i64>
  %results:2 = flow.dispatch.region -> (tensor<4xi64>, tensor<i64>) {
    %ascending:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %labels : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_label: i64, %rhs_label: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<4xi64>, tensor<4xi64>
    %descending:2 = iree_linalg_ext.sort dimension(0)
        outs(%ascending#0, %ascending#1 : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_label: i64, %rhs_label: i64):
      %cmp = arith.cmpi sge, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<4xi64>, tensor<4xi64>
    %sum = linalg.generic {indexing_maps = [#identity, #scalar], iterator_types = ["reduction"]}
        ins(%ascending#0 : tensor<4xi64>) outs(%sum_init : tensor<i64>) {
    ^bb0(%key: i64, %acc: i64):
      %position = linalg.index 0 : index
      %position_i64 = arith.index_cast %position : index to i64
      %one = arith.constant 1 : i64
      %weight = arith.addi %position_i64, %one : i64
      %weighted = arith.muli %weight, %key : i64
      %next = arith.addi %weighted, %acc : i64
      linalg.yield %next : i64
    } -> tensor<i64>
    flow.return %descending#1, %sum : tensor<4xi64>, tensor<i64>
  }
  return %results#0, %results#1, %keys, %labels
      : tensor<4xi64>, tensor<i64>, tensor<4xi64>, tensor<4xi64>
}
// EARLIER-LABEL: EXEC @two_sorts_keep_internal_version
// EARLIER: 4xi64=7 9 11 -3
// EARLIER: i64=30
// EARLIER: 4xi64=4 1 3 2
// EARLIER: 4xi64=7 -3 9 11
