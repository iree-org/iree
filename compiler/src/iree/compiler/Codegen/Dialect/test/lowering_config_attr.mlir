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
      lowering_config = #iree_codegen.lowering_config<tiling_levels = []>} {
    return
  }
}
// CHECK: #config = #iree_codegen.lowering_config<tiling_levels = []>

// -----

module {
  func.func @test() attributes {
      lowering_config = #iree_codegen.lowering_config<tiling_levels = [[], [10]], native_vector_size = [32, 32]>} {
    return
  }
}
// CHECK: #config = #iree_codegen.lowering_config<tiling_levels = {{\[}}[], [10]{{\]}}, native_vector_size = [32, 32]>

// -----

module {
  func.func @test() attributes {
     compilation_info = #iree_codegen.compilation_info<
         lowering_config = <tiling_levels = []>,
         translation_info = <CPUDefault>>} {
    return
  }
}
// CHECK: #config = #iree_codegen.lowering_config<tiling_levels = []>
// CHECK: #translation = #iree_codegen.translation_info<CPUDefault>
// CHECK: #compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>


// -----

module {
  func.func @test() attributes {
     compilation_info = #iree_codegen.compilation_info<
         lowering_config = <tiling_levels = []>,
         translation_info = <CPUDefault>,
         workgroup_size = [16, 4, 1],
         subgroup_size = 32>} {
    return
  }
}
// CHECK: #config = #iree_codegen.lowering_config<tiling_levels = []>
// CHECK: #translation = #iree_codegen.translation_info<CPUDefault>
// CHECK: #compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation, workgroup_size = [16, 4, 1], subgroup_size = 32>

// -----

module {
  func.func @test() attributes {
    export_config = #iree_codegen.export_config<workgroup_size = [4, 1]>
  } {
    return
  }
}
// CHECK: #iree_codegen.export_config<workgroup_size = [4, 1]
