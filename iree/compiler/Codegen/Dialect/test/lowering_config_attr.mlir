// RUN: iree-opt -split-input-file %s | FileCheck %s

module attributes {
  translation.info = #iree_codegen.translation.info<"CPUDefault", workload_per_wg = [32, 42]>
} { }
// CHECK: #translation = #iree_codegen.translation.info<"CPUDefault", workload_per_wg = [32, 42]>

// -----

module attributes {
  translation.info = #iree_codegen.translation.info<"CPUDefault", workload_per_wg = []>
} { }
// CHECK: #translation = #iree_codegen.translation.info<"CPUDefault", workload_per_wg = []>

// -----

module attributes {
  lowering.config = #iree_codegen.lowering.config<tile_sizes = [], native_vector_size = []>
} { }
// CHECK: #config = #iree_codegen.lowering.config<tile_sizes = [], native_vector_size = []>

// -----

module attributes {
  lowering.config = #iree_codegen.lowering.config<tile_sizes = [[], [10]], native_vector_size = [32, 32]>
} { }
// CHECK: #config = #iree_codegen.lowering.config<tile_sizes = {{\[}}[], [10]{{\]}}, native_vector_size = [32, 32]>

// -----

module attributes {
  compilation.info = #iree_codegen.compilation.info<
      #iree_codegen.lowering.config<tile_sizes = [], native_vector_size = []>,
      #iree_codegen.translation.info<"CPUDefault", workload_per_wg = []>,
      workgroup_size = []>
} { }
// CHECK: #compilation = #iree_codegen.compilation.info<<tile_sizes = [], native_vector_size = []>, <"CPUDefault", workload_per_wg = []>, workgroup_size = []>
