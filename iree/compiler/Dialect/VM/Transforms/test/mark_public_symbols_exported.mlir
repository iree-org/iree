// RUN: iree-opt -split-input-file -iree-vm-mark-public-symbols-exported %s | IreeFileCheck %s

// CHECK-LABEL: private @private_symbol
func private @private_symbol()

// CHECK-LABEL: @public_symbol
// CHECK-SAME: {iree.module.export}
func @public_symbol()
