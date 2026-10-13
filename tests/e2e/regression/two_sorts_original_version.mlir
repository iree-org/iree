// RUN: iree-run-mlir --Xcompiler,iree-hal-target-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --function=two_sorts_original_version --input="3xi64=[3 1 2]" | FileCheck %s
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --function=two_sorts_dependent --input="3xi64=[3 1 2]" --input="3xi64=[7 -3 9]" | FileCheck %s --check-prefix=DEPENDENT
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --function=sort_workspace_shape_chain --input="3xi64=[7 -3 9]" | FileCheck %s --check-prefix=LOCAL
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

// Only the terminal key version needs a hidden workspace export.
func.func @two_sorts_dependent(%keys: tensor<?xi64>, %labels: tensor<?xi64>)
    -> (tensor<?xi64>, tensor<?xi64>, tensor<?xi64>) {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %keys, %c0 : tensor<?xi64>
  %result = flow.dispatch.region -> (tensor<?xi64>{%n}) {
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
    flow.return %descending#1 : tensor<?xi64>
  }
  return %result, %keys, %labels : tensor<?xi64>, tensor<?xi64>, tensor<?xi64>
}
// DEPENDENT-LABEL: EXEC @two_sorts_dependent
// DEPENDENT: 3xi64=7 9 -3
// DEPENDENT: 3xi64=3 1 2
// DEPENDENT: 3xi64=7 -3 9

// A shape observation does not pin the tensor version across tied updates.
func.func @sort_workspace_shape_chain(%labels: tensor<?xi64>)
    -> (tensor<?xi64>, tensor<?xi64>, tensor<?xi64>) {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %labels, %c0 : tensor<?xi64>
  %results:2 = flow.dispatch.region -> (tensor<?xi64>{%n}, tensor<?xi64>{%n}) {
    %empty = tensor.empty(%n) : tensor<?xi64>
    %keys = linalg.generic {indexing_maps = [#identity], iterator_types = ["parallel"]}
        outs(%empty : tensor<?xi64>) {
    ^bb0(%out: i64):
      %zero = arith.constant 0 : i64
      %i = linalg.index 0 : index
      %value = arith.index_cast %i : index to i64
      %key = arith.subi %zero, %value : i64
      linalg.yield %key : i64
    } -> tensor<?xi64>
    %ascending:2 = iree_linalg_ext.sort dimension(0)
        outs(%keys, %labels : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_label: i64, %rhs_label: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    %size = tensor.dim %ascending#0, %c0 : tensor<?xi64>
    %shape_empty = tensor.empty(%size) : tensor<?xi64>
    %seven = arith.constant 7 : i64
    %filled = linalg.fill ins(%seven : i64) outs(%shape_empty : tensor<?xi64>) -> tensor<?xi64>
    %descending:2 = iree_linalg_ext.sort dimension(0)
        outs(%ascending#0, %ascending#1 : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_label: i64, %rhs_label: i64):
      %cmp = arith.cmpi sge, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    %final:2 = iree_linalg_ext.sort dimension(0)
        outs(%descending#0, %descending#1 : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_label: i64, %rhs_label: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    flow.return %final#1, %filled : tensor<?xi64>, tensor<?xi64>
  }
  return %results#0, %results#1, %labels : tensor<?xi64>, tensor<?xi64>, tensor<?xi64>
}
// LOCAL-LABEL: EXEC @sort_workspace_shape_chain
// LOCAL: 3xi64=9 -3 7
// LOCAL: 3xi64=7 7 7
// LOCAL: 3xi64=7 -3 9
