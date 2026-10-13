// RUN: iree-run-mlir --Xcompiler,iree-hal-target-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --function=argsort_internal_key_use --input="3xi64=[3 1 2]" | FileCheck %s --check-prefix=REDUCTION
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --function=argsort_internal_consumer --input="3xi64=[3 1 2]" | FileCheck %s --check-prefix=CONSUMER
// REDUCTION-LABEL: EXEC @argsort_internal_key_use
// REDUCTION: 3xi64=1 2 0
// REDUCTION: i64=6
// REDUCTION: 3xi64=3 1 2
// Internal data use still needs mutable keys even though no key tensor escapes.
#identity = affine_map<(d0) -> (d0)>
#scalar = affine_map<(d0) -> ()>
func.func @argsort_internal_key_use(%keys: tensor<?xi64>) -> (tensor<?xi64>, tensor<i64>, tensor<?xi64>) {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0 : i64
  %n = tensor.dim %keys, %c0 : tensor<?xi64>
  %empty = tensor.empty(%n) : tensor<?xi64>
  %sum_empty = tensor.empty() : tensor<i64>
  %sum_init = linalg.fill ins(%zero : i64) outs(%sum_empty : tensor<i64>) -> tensor<i64>
  %indices = linalg.generic {indexing_maps = [#identity], iterator_types = ["parallel"]} outs(%empty : tensor<?xi64>) {
  ^bb0(%out: i64):
    %i = linalg.index 0 : index
    %value = arith.index_cast %i : index to i64
    linalg.yield %value : i64
  } -> tensor<?xi64>
  %results:2 = flow.dispatch.region -> (tensor<?xi64>{%n}, tensor<i64>) {
    %sorted:2 = iree_linalg_ext.sort dimension(0) outs(%keys, %indices : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    %sum = linalg.generic {indexing_maps = [#identity, #scalar], iterator_types = ["reduction"]}
        ins(%sorted#0 : tensor<?xi64>) outs(%sum_init : tensor<i64>) {
    ^bb0(%key: i64, %acc: i64):
      %next = arith.addi %key, %acc : i64
      linalg.yield %next : i64
    } -> tensor<i64>
    flow.return %sorted#1, %sum : tensor<?xi64>, tensor<i64>
  }
  return %results#0, %results#1, %keys : tensor<?xi64>, tensor<i64>, tensor<?xi64>
}

// The consumer updates Sort's payload in place, while reading the original keys.
func.func @argsort_internal_consumer(%keys: tensor<?xi64>) -> (tensor<?xi64>, tensor<?xi64>) {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %keys, %c0 : tensor<?xi64>
  %empty = tensor.empty(%n) : tensor<?xi64>
  %indices = linalg.generic {indexing_maps = [#identity], iterator_types = ["parallel"]} outs(%empty : tensor<?xi64>) {
  ^bb0(%out: i64):
    %i = linalg.index 0 : index
    %value = arith.index_cast %i : index to i64
    linalg.yield %value : i64
  } -> tensor<?xi64>
  %result = flow.dispatch.region -> (tensor<?xi64>{%n}) {
    %sorted:2 = iree_linalg_ext.sort dimension(0) outs(%keys, %indices : tensor<?xi64>, tensor<?xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
      %cmp = arith.cmpi sle, %lhs, %rhs : i64
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi64>, tensor<?xi64>
    %combined = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel"]}
        ins(%keys : tensor<?xi64>) outs(%sorted#1 : tensor<?xi64>) {
    ^bb0(%original: i64, %index: i64):
      %value = arith.addi %original, %index : i64
      linalg.yield %value : i64
    } -> tensor<?xi64>
    flow.return %combined : tensor<?xi64>
  }
  return %result, %keys : tensor<?xi64>, tensor<?xi64>
}
// CONSUMER-LABEL: EXEC @argsort_internal_consumer
// CONSUMER: 3xi64=4 3 2
// CONSUMER: 3xi64=3 1 2
