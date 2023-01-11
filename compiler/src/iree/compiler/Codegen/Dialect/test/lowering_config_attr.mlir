// RUN: iree-opt --split-input-file %s | FileCheck %s

module {
  func.func @test() attributes {
      translation_info = #iree_codegen.translation_info<CPUDefault>} {
    return
  }
}
// CHECK: #translation = #iree_codegen.translation_info<CPUDefault>

// -----

module {
  func.func @test() attributes {
      lowering_config = #iree_codegen.lowering_config<tile_sizes = []>} {
    return
  }
}
// CHECK: #config = #iree_codegen.lowering_config<tile_sizes = []>

// -----

module {
  func.func @test() attributes {
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[], [10]], tile_interchange = [[], []], native_vector_size = [32, 32]>} {
    return
  }
}
// CHECK: #config = #iree_codegen.lowering_config<tile_sizes = {{\[}}[], [10]{{\]}}, tile_interchange = {{\[}}[], []], native_vector_size = [32, 32]>

// -----

module {
  func.func @test() attributes {
     compilation_info = #iree_codegen.compilation_info<
         lowering_config = <tile_sizes = []>,
         translation_info = <CPUDefault>,
         workgroup_size = []>} {
    return
  }
}
// CHECK: #compilation = #iree_codegen.compilation_info<lowering_config = <tile_sizes = []>, translation_info = <CPUDefault>>

// -----

module {
  func.func @test() attributes {
     compilation_info = #iree_codegen.compilation_info<
         lowering_config = <tile_sizes = []>,
         translation_info = <CPUDefault>,
         workgroup_size = [16, 4, 1],
         subgroup_size = 32>} {
    return
  }
}
// CHECK: #compilation = #iree_codegen.compilation_info<lowering_config = <tile_sizes = []>, translation_info = <CPUDefault>, workgroup_size = [16, 4, 1], subgroup_size = 32>
