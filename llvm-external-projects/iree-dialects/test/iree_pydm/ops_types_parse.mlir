// RUN: iree-dialects-opt %s | iree-dialects-opt | FileCheck --enable-var-scope --dump-input-filter=all %s

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

// CHECK-LABEL: @integer_types
// CHECK-SAME:     !iree_pydm.integer
// CHECK-SAME:     !iree_pydm.integer<32>
// CHECK-SAME:     !iree_pydm.integer<unsigned 32>
// CHECK-SAME:     !iree_pydm.integer<*>
iree_pydm.func private @integer_types(
    !iree_pydm.integer,
    !iree_pydm.integer<32>,
    !iree_pydm.integer<unsigned 32>,
    !iree_pydm.integer<*>)-> (!iree_pydm.exception_result, !iree_pydm.bool)

// CHECK-LABEL: @real_types
// CHECK-SAME:     !iree_pydm.real
// CHECK-SAME:     !iree_pydm.real<f32>
iree_pydm.func private @real_types(
    !iree_pydm.real,
    !iree_pydm.real<f32>)-> (!iree_pydm.exception_result, !iree_pydm.bool)

// CHECK-LABEL: @list_types
// CHECK-SAME:     !iree_pydm.list
// CHECK-SAME:     !iree_pydm.list<boxed,!iree_pydm.integer>
// CHECK-SAME:     !iree_pydm.list<unboxed,!iree_pydm.integer>
// CHECK-SAME:     !iree_pydm.list<empty,!iree_pydm.integer>
iree_pydm.func private @list_types(
    !iree_pydm.list,
    !iree_pydm.list<boxed,!iree_pydm.integer>,
    !iree_pydm.list<unboxed,!iree_pydm.integer>,
    !iree_pydm.list<empty,!iree_pydm.integer>)-> (!iree_pydm.exception_result, !iree_pydm.bool)
