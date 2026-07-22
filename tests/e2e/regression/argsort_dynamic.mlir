// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=generic -o %t.vmfb

// Regression test for https://github.com/iree-org/iree/issues/24735. The key
// result is unused, but sort still needs writable storage for the keys while it
// produces the permutation indices.

#identity = affine_map<(d0) -> (d0)>

func.func @argsort(%input: tensor<?xi64>) -> tensor<?xi64> {
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

  return %sorted#1 : tensor<?xi64>
}
