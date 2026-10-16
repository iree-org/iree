// RUN: iree-compile --iree-plugin=example --iree-example-flag \
// RUN:   --compile-to=input %s 2>&1 | FileCheck %s

// The linked copy; dynamic_link.mlir asserts the same of the loaded copy.

// CHECK: remark: This remark is from the example plugin activation (flag=1)
func.func @main() {
  return
}
