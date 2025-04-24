// RUN: iree-opt --pass-pipeline='builtin.module(util.func(iree-dispatch-creation-split-reduction-ops))' --iree-dispatch-creation-split-matmul-reduction=4 --split-input-file %s | FileCheck %s

#compilation = #iree_codegen.compilation_info<
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0]]>,
    translation_info = #iree_codegen.translation_info<pipeline = CPUDefault>>
util.func public @matmul(%arg0: tensor<100x200xf32>, %arg1: tensor<200x300xf32>, %arg2: tensor<100x300xf32>) -> tensor<100x300xf32> {
  %0 = linalg.matmul {compilation_info = #compilation}
    ins(%arg0, %arg1 : tensor<100x200xf32>, tensor<200x300xf32>)
    outs(%arg2 : tensor<100x300xf32>) -> tensor<100x300xf32>
  util.return %0 : tensor<100x300xf32>
}
// CHECK-DAG:   #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0]]>
// CHECK-DAG:   #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDefault>
// CHECK:       #[[INFO:.+]] = #iree_codegen.compilation_info<lowering_config = #[[CONFIG]], translation_info = #[[TRANSLATION]]>
// CHECK:       util.func public @matmul
// CHECK:         linalg.generic
// CHECK-SAME:      {compilation_info = #[[INFO]]}

// -----

#encoding = #iree_encoding.testing_encoding<>
util.func public @matmul_with_encoding(%arg0: tensor<100x200xf32, #encoding>, %arg1: tensor<200x300xf32>, %arg2: tensor<100x300xf32>) -> tensor<100x300xf32> {
  %0 = linalg.matmul
    ins(%arg0, %arg1 : tensor<100x200xf32, #encoding>, tensor<200x300xf32>)
    outs(%arg2 : tensor<100x300xf32>) -> tensor<100x300xf32>
  util.return %0 : tensor<100x300xf32>
}
// CHECK-LABEL: util.func public @matmul_with_encoding(
// CHECK:         linalg.matmul
