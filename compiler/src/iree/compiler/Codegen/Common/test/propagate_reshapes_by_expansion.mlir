// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-propagate-reshapes-by-expansion))" --split-input-file %s | FileCheck %s

func.func @reshape_and_lowering_config(%src: tensor<3x4xf16>, %dest: tensor<12xf16>, %dest2: tensor<12xf16>) -> tensor<12xf16> {
  %collapse = tensor.collapse_shape %src [[0, 1]] : tensor<3x4xf16> into tensor<12xf16>
  %copy = linalg.copy ins(%collapse : tensor<12xf16>) outs(%dest: tensor<12xf16>) -> tensor<12xf16>
  %copy2 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config} ins(%copy : tensor<12xf16>) outs(%dest2: tensor<12xf16>) -> tensor<12xf16>
  return %copy2: tensor<12xf16>
}

// CHECK-LABEL: func @reshape_and_lowering_config
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9]+]]: tensor<3x4xf16>
//       CHECK:   %[[COPY1:.+]] = linalg.generic {{.*}} ins(%[[SRC]]
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[COPY1]]
//       CHECK:   linalg.copy
//  CHECK-SAME:     lowering_config = #iree_gpu.derived_thread_config
//  CHECK-SAME:     ins(%[[COLLAPSE]]
