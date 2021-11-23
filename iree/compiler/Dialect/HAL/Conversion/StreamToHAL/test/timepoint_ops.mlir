// RUN: iree-opt -split-input-file -iree-hal-conversion %s | IreeFileCheck %s

// TODO(#7277): add new streams->hal conversion tests.
// CHECK: @todo
func @todo() {
  return
}
