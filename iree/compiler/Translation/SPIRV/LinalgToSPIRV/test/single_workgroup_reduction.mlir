// RUN: iree-opt -iree-xla-to-linalg-to-spirv %s | IreeFileCheck %s

// CHECK: spv.module
// CHECK: spv.globalVariable
// CHECK-SAME: built_in("LocalInvocationId")
// CHECK: spv.globalVariable
// CHECK-SAME: bind(0, 0)
// CHECK: spv.globalVariable
// CHECK-SAME: bind(0, 1)

// CHECK: capabilities = ["Shader", "GroupNonUniformArithmetic"]

// TODO(antiagainst): we are starting from linalg.generic directly and
// attaching a bunch of attributes on the function here to drive the
// CodeGen. Simplify this once we have the xla_hlo.reduce to linalg
// lowering.

module {
  func @simple_load_store(%input: memref<16xi32>, %output: memref<1xi32>)
  attributes  {
    iree.executable.export,
    iree.executable.workload = dense<[16, 1, 1]> : tensor<3xi32>,
    iree.executable.workgroup_size = dense<[16, 1, 1]> : tensor<3xi32>,
    iree.ordinal = 0 : i32,
    spv.entry_point_abi = {local_size = dense<[16, 1, 1]>: vector<3xi32>},
    spv.target_env = #spv.target_env<V_1_3, [], [Shader, GroupNonUniformArithmetic], {
      max_compute_workgroup_invocations = 128 : i32,
      max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>
    }>
  } {
    linalg.generic {
      args_in = 1, args_out = 1,
      iterator_types = ["reduction"],
      indexing_maps = [
        affine_map<(i) -> (i)>,
        affine_map<(i) -> (0)>
      ]
    } %input, %output {
      ^bb(%in: i32, %out: i32):
        %sum = addi %in, %out : i32
        linalg.yield %sum : i32
    } : memref<16xi32>, memref<1xi32>
    return
  }
}
