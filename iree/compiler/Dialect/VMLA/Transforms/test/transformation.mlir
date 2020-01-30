// RUN: iree-opt -split-input-file -iree-vmla-transformation-pipeline %s | IreeFileCheck %s

// CHECK-LABEL: @empty
func @empty() {
  // CHECK-NEXT: return
  return
}
