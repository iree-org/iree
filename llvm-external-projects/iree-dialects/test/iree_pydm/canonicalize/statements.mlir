// RUN: iree-dialects-opt --split-input-file --allow-unregistered-dialect --canonicalize %s | FileCheck  --dump-input-filter=all %s

// CHECK-LABEL: @elide_docstring
// CHECK-NOT: constant
// CHECK-NOT: expr_statement_discard
iree_pydm.func @elide_docstring(%arg0 : !iree_pydm.bool) -> (!iree_pydm.exception_result, !iree_pydm.bool) {
  %0 = constant "I AM A DOCSTRING" -> !iree_pydm.str
  expr_statement_discard %0 : !iree_pydm.str
  return %arg0 : !iree_pydm.bool
}
