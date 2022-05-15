// RUN: iree-dialects-opt --split-input-file --convert-iree-pydm-to-iree %s | FileCheck  --dump-input-filter=all %s

// CHECK-LABEL: @list_len
iree_pydm.func @list_len(%arg0 : !iree_pydm.list) -> (!iree_pydm.exception_result, !iree_pydm.integer) {
  // CHECK: %[[SIZE:.*]] = iree_input.list.size %arg0
  // CHECK: %[[SIZE_INT:.*]] = arith.index_cast %[[SIZE]] : index to i32
  // CHECK: return {{.*}}, %[[SIZE_INT]]
  %0 = len %arg0 : !iree_pydm.list -> <32>
  return %0 : !iree_pydm.integer<32>
}

// CHECK-LABEL: @tuple_len
iree_pydm.func @tuple_len(%arg0 : !iree_pydm.tuple) -> (!iree_pydm.exception_result, !iree_pydm.integer) {
  // CHECK: %[[SIZE:.*]] = iree_input.list.size %arg0
  // CHECK: %[[SIZE_INT:.*]] = arith.index_cast %[[SIZE]] : index to i32
  // CHECK: return {{.*}}, %[[SIZE_INT]]
  %0 = len %arg0 : !iree_pydm.tuple -> <32>
  return %0 : !iree_pydm.integer<32>
}
