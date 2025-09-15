// RUN: rm -rf %t && mkdir -p %t
// RUN: iree-compile --iree-hal-target-backends=rocm --iree-hip-target=gfx942 \
// RUN:   --iree-hip-emit-debug-info \
// RUN:   --iree-hal-dump-executable-binaries-to=%t \
// RUN:   %s -o %t/module.vmfb
// RUN: llvm-dwarfdump --debug-line %t/*.hsaco | FileCheck %s

// Test that the `--iree-hip-emit-debug-info` flag preserves debug information
// in the generated ELF binary by checking for specific source locations in DWARF.

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

// CHECK: .debug_line contents:
// CHECK: file_names[{{.*}}]:
// CHECK-NEXT: name: "test.mlir"
// ------ address         line column file ... flags
// CHECK: {{0x[0-9a-f]+}} 5    1      1 {{.*}} is_stmt
