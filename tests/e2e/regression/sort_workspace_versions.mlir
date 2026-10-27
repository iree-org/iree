// RUN: iree-run-mlir --Xcompiler,iree-hal-target-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --function=dependent_observed_dynamic --input="4xi64=[4 1 3 2]" --input="4xi64=[7 -3 9 11]" | FileCheck %s --check-prefix=VERSION
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --function=chained_output_inits --input="3xi64=[3 1 2]" --input="3xi64=[10 20 30]" | FileCheck %s --check-prefix=CHAIN

// The earlier key SSA version is returned after a dependent Sort updates it.
func.func @dependent_observed_dynamic(%keys: tensor<?xi64>, %labels: tensor<?xi64>)
    -> (tensor<?xi64>, tensor<?xi64>, tensor<?xi64>, tensor<?xi64>) {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %keys, %c0 : tensor<?xi64>
  %empty = tensor.empty(%n) : tensor<?xi64>
  %results:2 = flow.dispatch.region -> (tensor<?xi64>{%n}, tensor<?xi64>{%n}) {
    %ascending:2 = iree_linalg_ext.sort dimension(0)
        ins(%keys, %labels : tensor<?xi64>, tensor<?xi64>)
        outs(%empty, %empty : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_label: i64, %rhs_label: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    %descending:2 = iree_linalg_ext.sort dimension(0)
        ins(%ascending#0, %ascending#1 : tensor<?xi64>, tensor<?xi64>)
        outs(%empty, %empty : tensor<?xi64>, tensor<?xi64>) {
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

// A later Sort may reuse an earlier output as its destination without reading
// that earlier value. Both Sorts still require their own writable working state.
func.func @chained_output_inits(%keys: tensor<?xi64>, %labels: tensor<?xi64>) -> tensor<?xi64> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %keys, %c0 : tensor<?xi64>
  %empty = tensor.empty(%n) : tensor<?xi64>
  %result = flow.dispatch.region -> (tensor<?xi64>{%n}) {
    %first:2 = iree_linalg_ext.sort dimension(0)
        ins(%keys, %labels : tensor<?xi64>, tensor<?xi64>)
        outs(%empty, %empty : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %ll: i64, %rl: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    %second:2 = iree_linalg_ext.sort dimension(0)
        ins(%keys, %first#1 : tensor<?xi64>, tensor<?xi64>)
        outs(%first#0, %empty : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %ll: i64, %rl: i64):
      %cmp = arith.cmpi sge, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    flow.return %second#1 : tensor<?xi64>
  }
  return %result : tensor<?xi64>
}
// CHAIN-LABEL: EXEC @chained_output_inits
// CHAIN: 3xi64=20 10 30
