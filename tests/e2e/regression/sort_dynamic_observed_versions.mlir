// RUN: iree-run-mlir --Xcompiler,iree-hal-target-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --function=local_shared_scalar_observed_keys --input="3xi64=[7 -3 9]" | FileCheck %s --check-prefix=SCALAR
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --function=dependent_observed_dynamic --input="4xi64=[4 1 3 2]" --input="4xi64=[7 -3 9 11]" | FileCheck %s --check-prefix=VERSION

// The scalar feeding local keys is also needed by another live operation.
// Sort must leave the original key tensor and its shared initializer intact.
#identity = affine_map<(d0) -> (d0)>
#scalar = affine_map<(d0) -> ()>
func.func @local_shared_scalar_observed_keys(%labels: tensor<?xi64>)
    -> (tensor<?xi64>, tensor<?xi64>, tensor<?xi64>, tensor<?xi64>) {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %labels, %c0 : tensor<?xi64>
  %results:3 = flow.dispatch.region ->
      (tensor<?xi64>{%n}, tensor<?xi64>{%n}, tensor<?xi64>{%n}) {
    %zero = arith.constant 0 : i64
    %sum_empty = tensor.empty() : tensor<i64>
    %sum_init = linalg.fill ins(%zero : i64) outs(%sum_empty : tensor<i64>) -> tensor<i64>
    %sum = linalg.generic {indexing_maps = [#identity, #scalar], iterator_types = ["reduction"]}
        ins(%labels : tensor<?xi64>) outs(%sum_init : tensor<i64>) {
    ^bb0(%label: i64, %acc: i64):
      %next = arith.addi %label, %acc : i64
      linalg.yield %next : i64
    } -> tensor<i64>
    %seed = tensor.extract %sum[] : tensor<i64>
    %key_empty = tensor.empty(%n) : tensor<?xi64>
    %filled = linalg.fill ins(%seed : i64) outs(%key_empty : tensor<?xi64>) -> tensor<?xi64>
    %keys = linalg.generic {indexing_maps = [#identity], iterator_types = ["parallel"]}
        outs(%filled : tensor<?xi64>) {
    ^bb0(%base: i64):
      %i = linalg.index 0 : index
      %iv = arith.index_cast %i : index to i64
      %key = arith.subi %base, %iv : i64
      linalg.yield %key : i64
    } -> tensor<?xi64>
    %aux_empty = tensor.empty(%n) : tensor<?xi64>
    %aux = linalg.fill ins(%seed : i64) outs(%aux_empty : tensor<?xi64>) -> tensor<?xi64>
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %labels : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_label: i64, %rhs_label: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    flow.return %sorted#1, %keys, %aux : tensor<?xi64>, tensor<?xi64>, tensor<?xi64>
  }
  return %results#0, %results#1, %results#2, %labels
      : tensor<?xi64>, tensor<?xi64>, tensor<?xi64>, tensor<?xi64>
}
// SCALAR-LABEL: EXEC @local_shared_scalar_observed_keys
// SCALAR: 3xi64=9 -3 7
// SCALAR: 3xi64=13 12 11
// SCALAR: 3xi64=13 13 13
// SCALAR: 3xi64=7 -3 9

// The earlier key SSA version is returned after a dependent Sort updates it.
func.func @dependent_observed_dynamic(%keys: tensor<?xi64>, %labels: tensor<?xi64>)
    -> (tensor<?xi64>, tensor<?xi64>, tensor<?xi64>, tensor<?xi64>) {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %keys, %c0 : tensor<?xi64>
  %results:2 = flow.dispatch.region -> (tensor<?xi64>{%n}, tensor<?xi64>{%n}) {
    %ascending:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %labels : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_label: i64, %rhs_label: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    %descending:2 = iree_linalg_ext.sort dimension(0)
        outs(%ascending#0, %ascending#1 : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_label: i64, %rhs_label: i64):
      %cmp = arith.cmpi sge, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    flow.return %descending#1, %ascending#0 : tensor<?xi64>, tensor<?xi64>
  }
  return %results#0, %results#1, %keys, %labels
      : tensor<?xi64>, tensor<?xi64>, tensor<?xi64>, tensor<?xi64>
}
// VERSION-LABEL: EXEC @dependent_observed_dynamic
// VERSION: 4xi64=7 9 11 -3
// VERSION: 4xi64=1 2 3 4
// VERSION: 4xi64=4 1 3 2
// VERSION: 4xi64=7 -3 9 11
