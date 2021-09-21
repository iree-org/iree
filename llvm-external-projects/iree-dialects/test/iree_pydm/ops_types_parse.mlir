// RUN: iree-dialects-opt %s | iree-dialects-opt

iree_pydm.func @free_var(%arg0 : !iree_pydm.bool) -> (!iree_pydm.exception_result, !iree_pydm.bool) {
  %var = alloc_free_var "foo" -> !iree_pydm.free_var_ref
  store_var %var = %arg0 : !iree_pydm.free_var_ref, !iree_pydm.bool
  %0 = load_var %var : !iree_pydm.free_var_ref -> !iree_pydm.bool
  return %0 : !iree_pydm.bool
}

iree_pydm.func @free_var_index(%arg0 : !iree_pydm.bool) -> (!iree_pydm.exception_result, !iree_pydm.bool) {
  %var = alloc_free_var "foo"[1] -> !iree_pydm.free_var_ref
  store_var %var = %arg0 : !iree_pydm.free_var_ref, !iree_pydm.bool
  %0 = load_var %var : !iree_pydm.free_var_ref -> !iree_pydm.bool
  return %0 : !iree_pydm.bool
}
