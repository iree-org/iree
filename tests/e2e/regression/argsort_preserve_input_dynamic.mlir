// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --input="3xi64=[3 1 2]" | FileCheck %s --check-prefix=GENERAL
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --input="3xi64=[2 1 2]" | FileCheck %s --check-prefix=EQUAL
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --input="1xi64=[7]" | FileCheck %s --check-prefix=SINGLE
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=llvm-cpu --Xcompiler,iree-llvmcpu-target-cpu=generic %s --input="0xi64" | FileCheck %s --check-prefix=EMPTY

// The working key lane may be writable without changing the input or its view.
#identity = affine_map<(d0) -> (d0)>

func.func @argsort_preserve_input(%input: tensor<?xi64>)
    -> (tensor<?xi64>, tensor<?xi64>, tensor<1x?xi64>) {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %input, %c0 : tensor<?xi64>
  %indices_init = tensor.empty(%dim) : tensor<?xi64>
  %indices = linalg.generic {
      indexing_maps = [#identity], iterator_types = ["parallel"]}
      outs(%indices_init : tensor<?xi64>) {
  ^bb0(%out: i64):
    %index = linalg.index 0 : index
    %index_i64 = arith.index_cast %index : index to i64
    linalg.yield %index_i64 : i64
  } -> tensor<?xi64>

  %sorted:2 = iree_linalg_ext.sort dimension(0)
      outs(%input, %indices : tensor<?xi64>, tensor<?xi64>) {
  ^bb0(%lhs_key: i64, %rhs_key: i64, %lhs_index: i64, %rhs_index: i64):
    %take_lhs = arith.cmpi sle, %lhs_key, %rhs_key : i64
    iree_linalg_ext.yield %take_lhs : i1
  } -> tensor<?xi64>, tensor<?xi64>

  %alias = tensor.expand_shape %input [[0, 1]] output_shape [1, %dim]
      : tensor<?xi64> into tensor<1x?xi64>
  return %sorted#1, %input, %alias
      : tensor<?xi64>, tensor<?xi64>, tensor<1x?xi64>
}

// GENERAL-LABEL: EXEC @argsort_preserve_input
// GENERAL: 3xi64=1 2 0
// GENERAL: 3xi64=3 1 2
// GENERAL: 1x3xi64=[3 1 2]
// EQUAL-LABEL: EXEC @argsort_preserve_input
// EQUAL: 3xi64=1 0 2
// EQUAL: 3xi64=2 1 2
// EQUAL: 1x3xi64=[2 1 2]
// SINGLE-LABEL: EXEC @argsort_preserve_input
// SINGLE: 1xi64=0
// SINGLE: 1xi64=7
// SINGLE: 1x1xi64=[7]
// EMPTY-LABEL: EXEC @argsort_preserve_input
// EMPTY: 0xi64=
// EMPTY: 0xi64=
// EMPTY: 1x0xi64=[]
