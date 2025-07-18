// Passing iree-opt twice to test custom assembly format.
// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

func.func @test_empty_lowering_config() attributes {
    lowering_config = #iree_cpu.lowering_config<>} {
  return
}
// CHECK:       #[[$CONFIG:.+]] = #iree_cpu.lowering_config<>
// CHECK-LABEL: @test_empty_lowering_config()
// CHECK-SAME:    lowering_config = #[[$CONFIG]]

// -----

func.func @test_full_lowering_config() attributes {
    lowering_config = #iree_cpu.lowering_config<
      distribution = [128, 128, 0],
      cache_parallel = [64, 64, 0],
      cache_reduction = [0, 0, 16],
      vector_common_parallel = [4, 4, 0],
      vector_reduction = [0, 0, 4],
      vector_inner_parallel = [0, 0, 0]
    >} {
  return
}
// Order matters because it is sorted.
// CHECK:       #[[$CONFIG:.+]] = #iree_cpu.lowering_config<
// CHECK-SAME:      cache_parallel = [64, 64, 0]
// CHECK-SAME:      cache_reduction = [0, 0, 16]
// CHECK-SAME:      distribution = [128, 128, 0]
// CHECK-SAME:      vector_common_parallel = [4, 4, 0]
// CHECK-SAME:      vector_inner_parallel = [0, 0, 0]
// CHECK-SAME:      vector_reduction = [0, 0, 4]
// CHECK-LABEL: @test_full_lowering_config()
// CHECK-SAME:    lowering_config = #[[$CONFIG]]

// -----

func.func @test_full_lowering_config_with_scalable_vector() attributes {
    lowering_config = #iree_cpu.lowering_config<
      distribution = [128, 128, 0],
      cache_parallel = [64, 64, 0],
      cache_reduction = [0, 0, 16],
      vector_common_parallel = [[4], [4], 0],
      vector_reduction = [0, 0, [4]],
      vector_inner_parallel = [0, 0, 0]
    >} {
  return
}
// Order matters because it is sorted.
// CHECK:       #[[$CONFIG:.+]] = #iree_cpu.lowering_config<
// CHECK-SAME{LITERAL}:      cache_parallel = [64, 64, 0]
// CHECK-SAME{LITERAL}:      cache_reduction = [0, 0, 16]
// CHECK-SAME{LITERAL}:      distribution = [128, 128, 0]
// CHECK-SAME{LITERAL}:      vector_common_parallel = [[4], [4], 0]
// CHECK-SAME{LITERAL}:      vector_inner_parallel = [0, 0, 0]
// CHECK-SAME{LITERAL}:      vector_reduction = [0, 0, [4]]
// CHECK-LABEL:         @test_full_lowering_config_with_scalable_vector()
// CHECK-SAME:    lowering_config = #[[$CONFIG]]

// -----

func.func @test_arbitrary_keys() attributes {
    lowering_config = #iree_cpu.lowering_config<
      test = 123 : i32
    >} {
  return
}
// CHECK:       #[[$CONFIG:.+]] = #iree_cpu.lowering_config<
// CHECK-SAME:      test = 123 : i32
// CHECK-LABEL: @test_arbitrary_keys()
// CHECK-SAME:    lowering_config = #[[$CONFIG]]
