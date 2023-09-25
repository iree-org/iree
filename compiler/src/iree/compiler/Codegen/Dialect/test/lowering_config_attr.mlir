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
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[], [10]], native_vector_size = [32, 32]>} {
    return
  }
}
// CHECK: #config = #iree_codegen.lowering_config<tile_sizes = {{\[}}[], [10]{{\]}}, native_vector_size = [32, 32]>

// -----

module {
  func.func @test() attributes {
     compilation_info = #iree_codegen.compilation_info<
         lowering_config = <tile_sizes = []>,
         translation_info = <CPUDefault>>} {
    return
  }
}
// CHECK: #config = #iree_codegen.lowering_config<tile_sizes = []>
// CHECK: #translation = #iree_codegen.translation_info<CPUDefault>
// CHECK: #compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>


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
// CHECK: #config = #iree_codegen.lowering_config<tile_sizes = []>
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

// -----

module {
  /// Lowering config where the middle size of the second level is scalable.
  func.func @scalable_tile_sizes() attributes {
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 0], [1, [32], 0], [0, 0, 1], [0, 0, 0]]>} {
    return
  }
  // CHECK: #config = #iree_codegen.lowering_config<tile_sizes = {{\[}}[128, 128, 0], [1, [32], 0], [0, 0, 1], [0, 0, 0]{{\]}}>
}

// -----

module {
  /// Lowering config where the middle size of the second level is scalable has a tile interchange.
  func.func @scalable_tile_sizes() attributes {
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 0], {sizes=[1, [32], 0], interchange=[2, 1, 0]}, [0, 0, 1], [0, 0, 0]]>} {
    return
  }
  // CHECK: #config = #iree_codegen.lowering_config<tile_sizes = {{\[}}[128, 128, 0], {sizes = [1, [32], 0], interchange = [2, 1, 0]}, [0, 0, 1], [0, 0, 0]{{\]}}>
}
