// RUN: iree-compile --iree-plugin=example --iree-example-flag \
// RUN:   --compile-to=input %s 2>&1 | FileCheck %s

// The example plugin linked into the compiler. dynamic_link.mlir asserts the
// same behaviour for the same source built as a shared library, so the two
// files must keep the same CHECK.

// CHECK: remark: This remark is from the example plugin activation (flag=1)
func.func @main() {
  return
}
