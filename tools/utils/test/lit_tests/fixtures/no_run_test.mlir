// This test intentionally has NO RUN lines in header or body.
// CHECK-LABEL: @no_run
// CHECK: constant
func @no_run() {
  // CHECK: return
  return
}
