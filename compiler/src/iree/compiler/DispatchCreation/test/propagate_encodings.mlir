// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-propagate-encodings))" --split-input-file %s | FileCheck %s

// TODO(#21970): Add real tests. It is a workaround for an error: no check strings found with prefix CHECK
// CHECK: @foo
util.func @foo() {
  util.return
}
