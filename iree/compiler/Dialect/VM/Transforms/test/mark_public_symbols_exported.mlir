// RUN: iree-opt -split-input-file -iree-vm-mark-public-symbols-exported %s | IreeFileCheck %s

// CHECK-LABEL: @private_symbol
// CHECK-SAME: {sym_visibility = "private"}
func @private_symbol() attributes {sym_visibility = "private"}

// CHECK-LABEL: @public_symbol
// CHECK-SAME: {iree.module.export}
func @public_symbol()
