// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --function=argsort_trailing_float_keys --input="3xf32=[3.5 1.5 2.5]" | FileCheck %s --check-prefix=TRAILING
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --function=argsort_trailing_float_keys --input="3xf32=[2 1 2]" | FileCheck %s --check-prefix=EQUAL
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --function=argsort_shared_keys --input="3xi64=[3 1 2]" | FileCheck %s --check-prefix=SHARED
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --function=argsort_shared_keys --input="3xi64=[9223372036854775807 -9223372036854775808 0]" | FileCheck %s --check-prefix=EXTREMES
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --function=argsort_shared_keys --input="0xi64" | FileCheck %s --check-prefix=EMPTY
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --function=sort_carried_labels --input="2xi64=[2 1]" --input="2xi64=[1 0]" | FileCheck %s --check-prefix=LABELS
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --function=sort_carried_labels --input="2xi64=[2 1]" --input="2xi64=[7 -3]" | FileCheck %s --check-prefix=NONINDICES
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --function=sort_and_original --input="3xi64=[3 1 2]" | FileCheck %s --check-prefix=ORIGINAL
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --function=sort_and_original --input="0xi64" | FileCheck %s --check-prefix=ZERO

#identity = affine_map<(d0) -> (d0)>

func.func @argsort_trailing_float_keys(%input: tensor<?xf32>)
    -> tensor<?xi64> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %input, %c0 : tensor<?xf32>
  %empty = tensor.empty(%n) : tensor<?xi64>
  %indices = linalg.generic {
      indexing_maps = [#identity], iterator_types = ["parallel"]}
      outs(%empty : tensor<?xi64>) {
  ^bb0(%out: i64):
    %index = linalg.index 0 : index
    %value = arith.index_cast %index : index to i64
    linalg.yield %value : i64
  } -> tensor<?xi64>
  %keys_init = tensor.empty(%n) : tensor<?xf32>
  %sorted:2 = iree_linalg_ext.sort dimension(0)
      ins(%indices, %input : tensor<?xi64>, tensor<?xf32>)
      outs(%empty, %keys_init : tensor<?xi64>, tensor<?xf32>) {
  ^bb0(%lhs_index: i64, %rhs_index: i64, %lhs: f32, %rhs: f32):
    %cmp = arith.cmpf ole, %lhs, %rhs : f32
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<?xi64>, tensor<?xf32>
  return %sorted#0 : tensor<?xi64>
}

// Both key lanes need working state even though their source SSA value is equal.
func.func @argsort_shared_keys(%input: tensor<?xi64>)
    -> (tensor<?xi64>, tensor<?xi64>) {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %input, %c0 : tensor<?xi64>
  %empty = tensor.empty(%n) : tensor<?xi64>
  %indices = linalg.generic {
      indexing_maps = [#identity], iterator_types = ["parallel"]}
      outs(%empty : tensor<?xi64>) {
  ^bb0(%out: i64):
    %index = linalg.index 0 : index
    %value = arith.index_cast %index : index to i64
    linalg.yield %value : i64
  } -> tensor<?xi64>
  %sorted:3 = iree_linalg_ext.sort dimension(0)
      ins(%input, %input, %indices : tensor<?xi64>, tensor<?xi64>, tensor<?xi64>)
      outs(%empty, %empty, %empty : tensor<?xi64>, tensor<?xi64>, tensor<?xi64>) {
  ^bb0(%lhs0: i64, %rhs0: i64, %lhs1: i64, %rhs1: i64,
       %lhs_index: i64, %rhs_index: i64):
    %cmp0 = arith.cmpi sle, %lhs0, %rhs0 : i64
    %cmp1 = arith.cmpi sle, %lhs1, %rhs1 : i64
    %cmp = arith.andi %cmp0, %cmp1 : i1
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<?xi64>, tensor<?xi64>, tensor<?xi64>
  return %sorted#2, %input : tensor<?xi64>, tensor<?xi64>
}

// Carried values are labels; they need not identify positions in the keys.
func.func @sort_carried_labels(%keys: tensor<?xi64>, %labels: tensor<?xi64>)
    -> (tensor<?xi64>, tensor<?xi64>, tensor<?xi64>) {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %keys, %c0 : tensor<?xi64>
  %empty = tensor.empty(%n) : tensor<?xi64>
  %sorted:2 = iree_linalg_ext.sort dimension(0)
      ins(%keys, %labels : tensor<?xi64>, tensor<?xi64>)
      outs(%empty, %empty : tensor<?xi64>, tensor<?xi64>) {
  ^bb0(%lhs: i64, %rhs: i64, %lhs_label: i64, %rhs_label: i64):
    %cmp = arith.cmpi sle, %lhs, %rhs : i64
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<?xi64>, tensor<?xi64>
  return %sorted#1, %keys, %labels
      : tensor<?xi64>, tensor<?xi64>, tensor<?xi64>
}

// Sorting produces a new tensor while preserving the original input.
func.func @sort_and_original(%input: tensor<?xi64>)
    -> (tensor<?xi64>, tensor<?xi64>) {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %input, %c0 : tensor<?xi64>
  %empty = tensor.empty(%n) : tensor<?xi64>
  %sorted = iree_linalg_ext.sort dimension(0) ins(%input : tensor<?xi64>)
      outs(%empty : tensor<?xi64>) {
  ^bb0(%lhs: i64, %rhs: i64):
    %cmp = arith.cmpi sle, %lhs, %rhs : i64
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<?xi64>
  return %sorted, %input : tensor<?xi64>, tensor<?xi64>
}

// TRAILING-LABEL: EXEC @argsort_trailing_float_keys
// TRAILING: 3xi64=1 2 0
// EQUAL-LABEL: EXEC @argsort_trailing_float_keys
// EQUAL: 3xi64=1 0 2
// SHARED-LABEL: EXEC @argsort_shared_keys
// SHARED: 3xi64=1 2 0
// SHARED: 3xi64=3 1 2
// EXTREMES-LABEL: EXEC @argsort_shared_keys
// EXTREMES: 3xi64=1 2 0
// EXTREMES: 3xi64=9223372036854775807 -9223372036854775808 0
// EMPTY-LABEL: EXEC @argsort_shared_keys
// EMPTY: 0xi64=
// EMPTY: 0xi64=
// LABELS-LABEL: EXEC @sort_carried_labels
// LABELS: 2xi64=0 1
// LABELS: 2xi64=2 1
// LABELS: 2xi64=1 0
// NONINDICES-LABEL: EXEC @sort_carried_labels
// NONINDICES: 2xi64=-3 7
// NONINDICES: 2xi64=2 1
// NONINDICES: 2xi64=7 -3
// ORIGINAL-LABEL: EXEC @sort_and_original
// ORIGINAL: 3xi64=1 2 3
// ORIGINAL: 3xi64=3 1 2
// ZERO-LABEL: EXEC @sort_and_original
// ZERO: 0xi64=
// ZERO: 0xi64=
