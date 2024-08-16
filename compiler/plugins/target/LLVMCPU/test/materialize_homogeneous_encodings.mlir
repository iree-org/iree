// RUN: iree-opt --split-input-file --iree-hal-device-assignment-pipeline --iree-global-opt-materialize-homogeneous-encodings %s | FileCheck %s

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {target_triple = "x86_64-none-elf", cpu_features = "+avx512f"}>
#map = affine_map<()[s0, s1] -> (-s1 + (s1 ceildiv s0) * s0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map1, #map2, #map3], round_dims_to = array<i64: 16, 16, 16>>
#device_target_llvm_cpu = #hal.device.target<"local", [#executable_target_embedded_elf_x86_64_]> : !hal.device
module attributes {hal.device.targets = [#device_target_llvm_cpu]} {
  util.func public @lhs_encoding(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %3 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
    %4 = iree_encoding.unset_encoding %3 : tensor<?x?xf32, #encoding> -> tensor<?x?xf32>
    util.return %4 : tensor<?x?xf32>
  }
}
// CHECK-LABEL: util.func public @lhs_encoding
// CHECK:         tensor.pack
// CHECK:         tensor.unpack
