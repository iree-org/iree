// RUN: iree-opt -split-input-file -iree-hal-transformation-pipeline %s | IreeFileCheck %s

// CHECK-LABEL: @empty
func @empty() {
  // CHECK-NEXT: return
  return
}
