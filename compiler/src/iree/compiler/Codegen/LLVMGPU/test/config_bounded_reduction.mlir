// RUN: iree-opt --split-input-file --iree-gpu-test-target=sm_80 --pass-pipeline='builtin.module(iree-llvmgpu-select-lowering-strategy)' --mlir-print-local-scope "%s" | FileCheck "%s"
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 --pass-pipeline='builtin.module(iree-llvmgpu-select-lowering-strategy)' --mlir-print-local-scope "%s" | FileCheck "%s" --check-prefix=AMD
// RUN: iree-opt --split-input-file --iree-gpu-test-target=sm_80 --iree-codegen-llvmgpu-configuration-pipeline --iree-codegen-llvmgpu-nvvm-lowering-pipeline='include-llvm-lowering=false' "%s" | FileCheck "%s" --check-prefix=LOWER

// Bounds affect launch configuration only: tensor types and reduction tails
// must stay dynamic. Unbounded and very large dimensions retain the default.

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>
]>
// LOWER-LABEL: func.func @bounded_128()
// LOWER: %[[CMP128:.*]] = arith.cmpi slt, {{.*}} : vector<{{.*}}index>
// LOWER: %[[MASK128:.*]] = vector.shape_cast %[[CMP128]] : vector<{{.*}}i1> to vector<1x4xi1>
// LOWER: vector.transfer_read {{.*}}, %{{.*}}, %[[MASK128]] {in_bounds = [true, true]} : memref<16x?xf32, {{.*}}>, vector<1x4xf32>
// LOWER: iree_codegen.dispatch_config @bounded_128 workgroup_size = [32, 1, 1]
// AMD-LABEL: func.func @bounded_128()
// AMD-SAME: workgroup_size = [64, 1, 1] subgroup_size = 64
// CHECK-LABEL: func.func @bounded_128()
// CHECK-SAME: translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [32, 1, 1] subgroup_size = 32
func.func @bounded_128() {
  %zero = arith.constant 0.0 : f32
  %arg = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %size = arith.index_castui %arg : i32 to index
  %n = util.assume.int %size<umin = 0, umax = 128> : index
  %source = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x?xf32>>{%n}
  %dest = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  %input = iree_tensor_ext.dispatch.tensor.load %source, offsets = [0, 0], sizes = [16, %n], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x?xf32>>{%n} -> tensor<16x?xf32>
  %empty = tensor.empty() : tensor<16xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<16xf32>) -> tensor<16xf32>
  // CHECK: linalg.generic
  // CHECK: ins(%{{.*}} : tensor<16x?xf32>)
  %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%input : tensor<16x?xf32>) outs(%init : tensor<16xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %v = arith.addf %x, %acc : f32
      linalg.yield %v : f32
    } -> tensor<16xf32>
  iree_tensor_ext.dispatch.tensor.store %sum, %dest, offsets = [0], sizes = [16], strides = [1] : tensor<16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>
]>
// LOWER-LABEL: func.func @bounded_130()
// LOWER: %[[CMP130:.*]] = arith.cmpi slt, {{.*}} : vector<{{.*}}index>
// LOWER: %[[MASK130:.*]] = vector.shape_cast %[[CMP130]] : vector<{{.*}}i1> to vector<1x4xi1>
// LOWER: vector.transfer_read {{.*}}, %{{.*}}, %[[MASK130]] {in_bounds = [true, true]} : memref<16x?xf32, {{.*}}>, vector<1x4xf32>
// LOWER: iree_codegen.dispatch_config @bounded_130 workgroup_size = [64, 1, 1]
// AMD-LABEL: func.func @bounded_130()
// AMD-SAME: workgroup_size = [64, 1, 1] subgroup_size = 64
// CHECK-LABEL: func.func @bounded_130()
// CHECK-SAME: translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [64, 1, 1] subgroup_size = 32
func.func @bounded_130() {
  %zero = arith.constant 0.0 : f32
  %arg = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %size = arith.index_castui %arg : i32 to index
  %n = util.assume.int %size<umin = 0, umax = 130> : index
  %source = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x?xf32>>{%n}
  %dest = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  %input = iree_tensor_ext.dispatch.tensor.load %source, offsets = [0, 0], sizes = [16, %n], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x?xf32>>{%n} -> tensor<16x?xf32>
  %empty = tensor.empty() : tensor<16xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<16xf32>) -> tensor<16xf32>
  // CHECK: linalg.generic
  // CHECK: ins(%{{.*}} : tensor<16x?xf32>)
  %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%input : tensor<16x?xf32>) outs(%init : tensor<16xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %v = arith.addf %x, %acc : f32
      linalg.yield %v : f32
    } -> tensor<16xf32>
  iree_tensor_ext.dispatch.tensor.store %sum, %dest, offsets = [0], sizes = [16], strides = [1] : tensor<16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>
]>
// LOWER-LABEL: func.func @bounded_one()
// LOWER: %[[CMPONE:.*]] = arith.cmpi slt, {{.*}} : vector<{{.*}}index>
// LOWER: %[[MASKONE:.*]] = vector.shape_cast %[[CMPONE]] : vector<{{.*}}i1> to vector<1x4xi1>
// LOWER: vector.transfer_read {{.*}}, %{{.*}}, %[[MASKONE]] {in_bounds = [true, true]} : memref<16x?xf32, {{.*}}>, vector<1x4xf32>
// LOWER: iree_codegen.dispatch_config @bounded_one workgroup_size = [32, 1, 1]
// AMD-LABEL: func.func @bounded_one()
// AMD-SAME: workgroup_size = [64, 1, 1] subgroup_size = 64
// CHECK-LABEL: func.func @bounded_one()
// CHECK-SAME: translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [32, 1, 1] subgroup_size = 32
func.func @bounded_one() {
  %zero = arith.constant 0.0 : f32
  %arg = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %size = arith.index_castui %arg : i32 to index
  %n = util.assume.int %size<umin = 0, umax = 1> : index
  %source = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x?xf32>>{%n}
  %dest = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  %input = iree_tensor_ext.dispatch.tensor.load %source, offsets = [0, 0], sizes = [16, %n], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x?xf32>>{%n} -> tensor<16x?xf32>
  %empty = tensor.empty() : tensor<16xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<16xf32>) -> tensor<16xf32>
  // CHECK: linalg.generic
  // CHECK: ins(%{{.*}} : tensor<16x?xf32>)
  %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%input : tensor<16x?xf32>) outs(%init : tensor<16xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %v = arith.addf %x, %acc : f32
      linalg.yield %v : f32
    } -> tensor<16xf32>
  iree_tensor_ext.dispatch.tensor.store %sum, %dest, offsets = [0], sizes = [16], strides = [1] : tensor<16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>
]>
// AMD-LABEL: func.func @unbounded()
// AMD-SAME: workgroup_size = [1024, 1, 1] subgroup_size = 64
// CHECK-LABEL: func.func @unbounded()
// CHECK-SAME: translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [1024, 1, 1] subgroup_size = 32
func.func @unbounded() {
  %zero = arith.constant 0.0 : f32
  %arg = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %size = arith.index_castui %arg : i32 to index
  %c0 = arith.constant 0 : index
  %n = arith.addi %size, %c0 : index
  %source = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x?xf32>>{%n}
  %dest = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  %input = iree_tensor_ext.dispatch.tensor.load %source, offsets = [0, 0], sizes = [16, %n], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x?xf32>>{%n} -> tensor<16x?xf32>
  %empty = tensor.empty() : tensor<16xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<16xf32>) -> tensor<16xf32>
  // CHECK: linalg.generic
  // CHECK: ins(%{{.*}} : tensor<16x?xf32>)
  %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%input : tensor<16x?xf32>) outs(%init : tensor<16xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %v = arith.addf %x, %acc : f32
      linalg.yield %v : f32
    } -> tensor<16xf32>
  iree_tensor_ext.dispatch.tensor.store %sum, %dest, offsets = [0], sizes = [16], strides = [1] : tensor<16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>
]>
// AMD-LABEL: func.func @zero_bound()
// AMD-SAME: workgroup_size = [1024, 1, 1] subgroup_size = 64
// CHECK-LABEL: func.func @zero_bound()
// CHECK-SAME: translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [1024, 1, 1] subgroup_size = 32
func.func @zero_bound() {
  %zero = arith.constant 0.0 : f32
  %arg = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %size = arith.index_castui %arg : i32 to index
  %n = util.assume.int %size<umin = 0, umax = 0> : index
  %source = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x?xf32>>{%n}
  %dest = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  %input = iree_tensor_ext.dispatch.tensor.load %source, offsets = [0, 0], sizes = [16, %n], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x?xf32>>{%n} -> tensor<16x?xf32>
  %empty = tensor.empty() : tensor<16xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<16xf32>) -> tensor<16xf32>
  // CHECK: linalg.generic
  // CHECK: ins(%{{.*}} : tensor<16x?xf32>)
  %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%input : tensor<16x?xf32>) outs(%init : tensor<16xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %v = arith.addf %x, %acc : f32
      linalg.yield %v : f32
    } -> tensor<16xf32>
  iree_tensor_ext.dispatch.tensor.store %sum, %dest, offsets = [0], sizes = [16], strides = [1] : tensor<16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>
]>
// AMD-LABEL: func.func @large_bound()
// AMD-SAME: workgroup_size = [1024, 1, 1] subgroup_size = 64
// CHECK-LABEL: func.func @large_bound()
// CHECK-SAME: translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [1024, 1, 1] subgroup_size = 32
func.func @large_bound() {
  %zero = arith.constant 0.0 : f32
  %arg = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %size = arith.index_castui %arg : i32 to index
  %n = util.assume.int %size<umax = 4294967296> : index
  %source = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x?xf32>>{%n}
  %dest = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  %input = iree_tensor_ext.dispatch.tensor.load %source, offsets = [0, 0], sizes = [16, %n], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x?xf32>>{%n} -> tensor<16x?xf32>
  %empty = tensor.empty() : tensor<16xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<16xf32>) -> tensor<16xf32>
  // CHECK: linalg.generic
  // CHECK: ins(%{{.*}} : tensor<16x?xf32>)
  %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%input : tensor<16x?xf32>) outs(%init : tensor<16xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %v = arith.addf %x, %acc : f32
      linalg.yield %v : f32
    } -> tensor<16xf32>
  iree_tensor_ext.dispatch.tensor.store %sum, %dest, offsets = [0], sizes = [16], strides = [1] : tensor<16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>
]>
// AMD-LABEL: func.func @union_bound()
// AMD-SAME: workgroup_size = [128, 1, 1] subgroup_size = 64
// CHECK-LABEL: func.func @union_bound()
// CHECK-SAME: translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [128, 1, 1] subgroup_size = 32
func.func @union_bound() {
  %zero = arith.constant 0.0 : f32
  %arg = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %size = arith.index_castui %arg : i32 to index
  %n = util.assume.int %size[<umax = 64>, <umax = 512>] : index
  %source = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x?xf32>>{%n}
  %dest = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  %input = iree_tensor_ext.dispatch.tensor.load %source, offsets = [0, 0], sizes = [16, %n], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x?xf32>>{%n} -> tensor<16x?xf32>
  %empty = tensor.empty() : tensor<16xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<16xf32>) -> tensor<16xf32>
  // CHECK: linalg.generic
  // CHECK: ins(%{{.*}} : tensor<16x?xf32>)
  %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%input : tensor<16x?xf32>) outs(%init : tensor<16xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %v = arith.addf %x, %acc : f32
      linalg.yield %v : f32
    } -> tensor<16xf32>
  iree_tensor_ext.dispatch.tensor.store %sum, %dest, offsets = [0], sizes = [16], strides = [1] : tensor<16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>
]>
// AMD-LABEL: func.func @computed_bound()
// AMD-SAME: workgroup_size = [64, 1, 1] subgroup_size = 64
// CHECK-LABEL: func.func @computed_bound()
// CHECK-SAME: translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [32, 1, 1] subgroup_size = 32
func.func @computed_bound() {
  %zero = arith.constant 0.0 : f32
  %arg = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %size = arith.index_castui %arg : i32 to index
  %bounded = util.assume.int %size<umax = 40> : index
  %c3 = arith.constant 3 : index
  %n = arith.muli %bounded, %c3 : index
  %source = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x?xf32>>{%n}
  %dest = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  %input = iree_tensor_ext.dispatch.tensor.load %source, offsets = [0, 0], sizes = [16, %n], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x?xf32>>{%n} -> tensor<16x?xf32>
  %empty = tensor.empty() : tensor<16xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<16xf32>) -> tensor<16xf32>
  // CHECK: linalg.generic
  // CHECK: ins(%{{.*}} : tensor<16x?xf32>)
  %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%input : tensor<16x?xf32>) outs(%init : tensor<16xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %v = arith.addf %x, %acc : f32
      linalg.yield %v : f32
    } -> tensor<16xf32>
  iree_tensor_ext.dispatch.tensor.store %sum, %dest, offsets = [0], sizes = [16], strides = [1] : tensor<16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>
]>
// Two independent dynamic dimensions separated by a static dimension. Only the
// last dimension is reduced; its bound must not be confused with the first.
// CHECK-LABEL: func.func @independent_parallel_and_reduction_dims()
// CHECK-SAME: workgroup_size = [64, 1, 1] subgroup_size = 32
// AMD-LABEL: func.func @independent_parallel_and_reduction_dims()
// AMD-SAME: workgroup_size = [64, 1, 1] subgroup_size = 64
func.func @independent_parallel_and_reduction_dims() {
  %zero = arith.constant 0.0 : f32
  %m_arg = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %n_arg = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %m_size = arith.index_castui %m_arg : i32 to index
  %n_size = arith.index_castui %n_arg : i32 to index
  %bounded_m = util.assume.int %m_size<umin = 0, umax = 1> : index
  %m = iree_tensor_ext.dispatch.workload.ordinal %bounded_m, 0 : index
  %n = util.assume.int %n_size<umin = 0, umax = 130> : index
  %source = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x4x?xf32>>{%m, %n}
  %dest = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x4xf32>>{%m}
  %input = iree_tensor_ext.dispatch.tensor.load %source, offsets = [0, 0, 0], sizes = [%m, 4, %n], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x4x?xf32>>{%m, %n} -> tensor<?x4x?xf32>
  %empty = tensor.empty(%m) : tensor<?x4xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x4xf32>) -> tensor<?x4xf32>
  %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%input : tensor<?x4x?xf32>) outs(%init : tensor<?x4xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %v = arith.addf %x, %acc : f32
      linalg.yield %v : f32
    } -> tensor<?x4xf32>
  iree_tensor_ext.dispatch.tensor.store %sum, %dest, offsets = [0, 0], sizes = [%m, 4], strides = [1, 1] : tensor<?x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x4xf32>>{%m}
  return
}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>
]>
// Multiple dynamic reduction axes conservatively keep the original estimate.
// Use independent bounds so neither axis is mistaken for the total reduction.
// CHECK-LABEL: func.func @multiple_reduction_dims()
// CHECK-SAME: workgroup_size = [1024, 1, 1] subgroup_size = 32
// AMD-LABEL: func.func @multiple_reduction_dims()
// AMD-SAME: workgroup_size = [1024, 1, 1] subgroup_size = 64
func.func @multiple_reduction_dims() {
  %zero = arith.constant 0.0 : f32
  %arg = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %size = arith.index_castui %arg : i32 to index
  %n = util.assume.int %size<umin = 0, umax = 128> : index
  %other_arg = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %other_size = arith.index_castui %other_arg : i32 to index
  %m = util.assume.int %other_size<umin = 0, umax = 2> : index
  %source = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x?x?xf32>>{%m, %n}
  %dest = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  %input = iree_tensor_ext.dispatch.tensor.load %source, offsets = [0, 0, 0], sizes = [16, %m, %n], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x?x?xf32>>{%m, %n} -> tensor<16x?x?xf32>
  %empty = tensor.empty() : tensor<16xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<16xf32>) -> tensor<16xf32>
  %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0)>],
      iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%input : tensor<16x?x?xf32>) outs(%init : tensor<16xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %v = arith.addf %x, %acc : f32
      linalg.yield %v : f32
    } -> tensor<16xf32>
  iree_tensor_ext.dispatch.tensor.store %sum, %dest, offsets = [0], sizes = [16], strides = [1] : tensor<16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  return
}
