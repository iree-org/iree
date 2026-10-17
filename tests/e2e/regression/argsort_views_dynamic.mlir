// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --function=argsort_collapsed_keys --input="1x3xi64=[3 1 2]" | FileCheck %s --check-prefix=COLLAPSED
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --function=argsort_collapsed_keys --input="1x0xi64" | FileCheck %s --check-prefix=EMPTY
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --function=argsort_rows --input="2x3xi64=[3 1 2][2 1 2]" | FileCheck %s --check-prefix=ROWS
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic --Xcompiler,iree-llvmcpu-distribution-size=1 %s --function=argsort_rows --input="2x3xi64=[3 1 2][2 1 2]" | FileCheck %s --check-prefix=ROWS

#identity = affine_map<(d0) -> (d0)>
#identity2 = affine_map<(d0, d1) -> (d0, d1)>

func.func @argsort_collapsed_keys(%input: tensor<1x?xi64>)
    -> (tensor<?xi64>, tensor<1x?xi64>) {
  %c0 = arith.constant 0 : index
  %keys = tensor.collapse_shape %input [[0, 1]]
      : tensor<1x?xi64> into tensor<?xi64>
  %n = tensor.dim %keys, %c0 : tensor<?xi64>
  %empty = tensor.empty(%n) : tensor<?xi64>
  %indices = linalg.generic {
      indexing_maps = [#identity], iterator_types = ["parallel"]}
      outs(%empty : tensor<?xi64>) {
  ^bb0(%out: i64):
    %index = linalg.index 0 : index
    %value = arith.index_cast %index : index to i64
    linalg.yield %value : i64
  } -> tensor<?xi64>
  %sorted:2 = iree_linalg_ext.sort dimension(0)
      outs(%keys, %indices : tensor<?xi64>, tensor<?xi64>) {
  ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
    %cmp = arith.cmpi sle, %lhs, %rhs : i64
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<?xi64>, tensor<?xi64>
  return %sorted#1, %input : tensor<?xi64>, tensor<1x?xi64>
}

func.func @argsort_rows(%input: tensor<?x?xi64>)
    -> (tensor<?x?xi64>, tensor<?x?xi64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %rows = tensor.dim %input, %c0 : tensor<?x?xi64>
  %cols = tensor.dim %input, %c1 : tensor<?x?xi64>
  %empty = tensor.empty(%rows, %cols) : tensor<?x?xi64>
  %indices = linalg.generic {
      indexing_maps = [#identity2],
      iterator_types = ["parallel", "parallel"]}
      outs(%empty : tensor<?x?xi64>) {
  ^bb0(%out: i64):
    %index = linalg.index 1 : index
    %value = arith.index_cast %index : index to i64
    linalg.yield %value : i64
  } -> tensor<?x?xi64>
  %sorted:2 = iree_linalg_ext.sort dimension(1)
      outs(%input, %indices : tensor<?x?xi64>, tensor<?x?xi64>) {
  ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):
    %cmp = arith.cmpi sle, %lhs, %rhs : i64
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<?x?xi64>, tensor<?x?xi64>
  return %sorted#1, %input : tensor<?x?xi64>, tensor<?x?xi64>
}

// COLLAPSED-LABEL: EXEC @argsort_collapsed_keys
// COLLAPSED: 3xi64=1 2 0
// COLLAPSED: 1x3xi64=[3 1 2]
// EMPTY-LABEL: EXEC @argsort_collapsed_keys
// EMPTY: 0xi64=
// EMPTY: 1x0xi64=[]
// ROWS-LABEL: EXEC @argsort_rows
// ROWS: 2x3xi64=[1 2 0][1 0 2]
// ROWS: 2x3xi64=[3 1 2][2 1 2]
