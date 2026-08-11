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
// CHECK-SAME:      cache_parallel = [64, 64, 0]
// CHECK-SAME:      cache_reduction = [0, 0, 16]
// CHECK-SAME:      distribution = [128, 128, 0]
// CHECK-SAME{LITERAL}:      vector_common_parallel = [[4], [4], 0]
// CHECK-SAME:      vector_inner_parallel = [0, 0, 0]
// CHECK-SAME{LITERAL}:      vector_reduction = [0, 0, [4]]
// CHECK-LABEL: @test_full_lowering_config_with_scalable_vector()
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

// -----

// Round-trip the LLVMCPU ukernel provider attribute. This is the CPU analogue
// of #rocm.ukernel_provider, set on a `hal.executable.target` config to enable
// the built-in C-bitcode ukernels under
// compiler/plugins/target/LLVMCPU/builtins/ukernel/.
func.func @test_ukernel_provider() attributes {
    iree_codegen.ukernel_provider = #iree_cpu.ukernel_provider} {
  return
}
// CHECK-LABEL: @test_ukernel_provider()
// CHECK-SAME:    iree_codegen.ukernel_provider = #iree_cpu.ukernel_provider

// -----

// Round-trip the inner-tile alignment hint attribute. It records, per tiling
// level, the InnerTileAlignment (Unknown|Multiple|Equal) of each
// iteration-domain dimension.
func.func @test_inner_tile_alignments() attributes {
    inner_tile_alignments =
        #iree_cpu.inner_tile_alignments<vector_common_parallel = [Unknown, Equal]>} {
  return
}
// CHECK-LABEL: @test_inner_tile_alignments()
// CHECK-SAME:    inner_tile_alignments = #iree_cpu.inner_tile_alignments<
// CHECK-SAME:      vector_common_parallel = [Unknown, Equal]>

// -----

// Multiple levels and all three alignment kinds, each level carrying one entry
// per iteration-domain dimension (here 2). Levels are printed sorted by name,
// regardless of input order.
func.func @test_inner_tile_alignments_multi_level() attributes {
    inner_tile_alignments = #iree_cpu.inner_tile_alignments<
      vector_inner_parallel = [Equal, Unknown],
      vector_common_parallel = [Multiple, Unknown],
      distribution = [Unknown, Equal]>} {
  return
}
// CHECK-LABEL: @test_inner_tile_alignments_multi_level()
// CHECK-SAME:    inner_tile_alignments = #iree_cpu.inner_tile_alignments<
// CHECK-SAME:      distribution = [Unknown, Equal],
// CHECK-SAME:      vector_common_parallel = [Multiple, Unknown],
// CHECK-SAME:      vector_inner_parallel = [Equal, Unknown]>
