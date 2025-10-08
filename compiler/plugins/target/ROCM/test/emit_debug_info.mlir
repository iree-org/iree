// RUN: iree-opt --mlir-print-debuginfo %s --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-codegen-llvmgpu-configuration-pipeline), iree-codegen-linalg-to-rocdl-pipeline{preserve-debug-info})))" | FileCheck %s

// Test that LLVM debug info attributes pass through without being stripped when
// the pipeline `preserve-debug-info` option is used

#di_file = #llvm.di_file<"test.mlir" in "/work">
#di_compile_unit = #llvm.di_compile_unit<
  id = distinct[0]<>,
  sourceLanguage = DW_LANG_C,
  file = #di_file,
  producer = "test",
  isOptimized = false,
  emissionKind = LineTablesOnly
>
#di_subprogram = #llvm.di_subprogram<
  id = distinct[1]<>,
  compileUnit = #di_compile_unit,
  scope = #di_file,
  name = "simple_mul",
  file = #di_file,
  line = 5,
  scopeLine = 5,
  subprogramFlags = Definition
>

module attributes {
  llvm.di_compile_unit = #di_compile_unit
} {
  func.func @simple_mul(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %0 = tensor.empty() : tensor<16xf32>
    %1 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%arg0 : tensor<16xf32>) outs(%0 : tensor<16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.mulf %in, %in : f32
      linalg.yield %2 : f32
    } -> tensor<16xf32>
    return %1 : tensor<16xf32>
  } loc(fused<#di_subprogram>["test.mlir":5:1])
}

// CHECK-DAG: [[DI_FILE:#[a-z_0-9]+]] = #llvm.di_file<"test.mlir" in "/work">
// CHECK-DAG: [[DI_CU:#[a-z_0-9]+]] = #llvm.di_compile_unit<{{.*}}file = [[DI_FILE]]{{.*}}>
// CHECK-DAG: module attributes {llvm.di_compile_unit = [[DI_CU]]}
// CHECK-DAG: [[DI_SP:#[a-z_0-9]+]] = #llvm.di_subprogram<{{.*}}compileUnit = [[DI_CU]]{{.*}}name = "simple_mul"{{.*}}line = 5{{.*}}>
// CHECK-DAG: [[LOC:#loc[0-9]+]] = loc("test.mlir":5:1)
