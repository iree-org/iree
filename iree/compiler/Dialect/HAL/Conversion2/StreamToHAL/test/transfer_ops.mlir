// RUN: iree-opt -split-input-file -iree-hal-conversion %s | IreeFileCheck %s

// TODO(#7277): swap HAL Conversion2->Conversion and add tests.
// CHECK: @todo
func @todo() {
  return
}
