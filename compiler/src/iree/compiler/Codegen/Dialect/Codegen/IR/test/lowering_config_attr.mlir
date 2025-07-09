// RUN: iree-opt --split-input-file --verify-diagnostics %s | FileCheck %s

module {
  func.func @test() attributes {
      translation_info = #iree_codegen.translation_info<pipeline = CPUDefault>} {
    return
  }
}
// CHECK: #translation = #iree_codegen.translation_info<pipeline = CPUDefault>

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
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[], [10]]>} {
    return
  }
}
// CHECK: #config = #iree_codegen.lowering_config<tile_sizes = {{\[}}[], [10]{{\]}}>

// -----

module {
  func.func @test() attributes {
     compilation_info = #iree_codegen.compilation_info<
         lowering_config = #iree_codegen.lowering_config<tile_sizes = []>,
         translation_info = #iree_codegen.translation_info<pipeline = CPUDefault>>} {
    return
  }
}
// CHECK: #config = #iree_codegen.lowering_config<tile_sizes = []>
// CHECK: #translation = #iree_codegen.translation_info<pipeline = CPUDefault>
// CHECK: #compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>


// -----

module {
  func.func @test() attributes {
     compilation_info = #iree_codegen.compilation_info<
         lowering_config = #iree_codegen.lowering_config<tile_sizes = []>,
         translation_info = #iree_codegen.translation_info<pipeline = CPUDefault workgroup_size = [16, 4, 1] subgroup_size = 32>>} {
    return
  }
}
// CHECK: #config = #iree_codegen.lowering_config<tile_sizes = []>
// CHECK: #translation = #iree_codegen.translation_info<pipeline = CPUDefault workgroup_size = [16, 4, 1] subgroup_size = 32>
// CHECK: #compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>

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

// -----

module {
  /// translation info cannot have more than 3 entries for workgroup size
  func.func @workgroup_size_more_than_3_err() attributes {
    // expected-error @+1 {{workgroup size cannot have more than 3 entries}}
    translation_info = #iree_codegen.translation_info<pipeline = None workgroup_size = [4, 1, 1, 1]> {
      return
    }
  }
}

// -----

module {
  /// translation info workgroup_size values needs to have non-negative values.
  func.func @workgroup_size_neg_err() attributes {
    // expected-error @+1 {{workgroup size value has to be greater than zero}}
    translation_info = #iree_codegen.translation_info<pipeline = None workgroup_size = [4, -1, 1]> {
      return
    }
  }
}

// -----

module {
  /// translation info workgroup_size values needs to have non-negative values.
  func.func @subgroup_size_neg_err() attributes {
    // expected-error @+1 {{subgroup size value cannot be negative}}
    translation_info = #iree_codegen.translation_info<pipeline = None subgroup_size = -1> {
      return
    }
  }
}
