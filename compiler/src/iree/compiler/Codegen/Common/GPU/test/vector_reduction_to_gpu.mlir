// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-vector-reduction-to-gpu, cse)))))' %s | FileCheck %s

#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @simple_reduce  {
  hal.executable.variant @cuda target(#executable_target_cuda_nvptx_fb) {
    hal.executable.export @simple_reduce layout(#pipeline_layout) attributes {
      workgroup_size = [32 : index, 1 : index, 1 : index]
    }
    builtin.module {
    func.func @simple_reduce() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
      %cst_0 = arith.constant 0.000000e+00 : f32
      %cst_1 = arith.constant dense<3.840000e+02> : vector<1xf32>
      %c32 = arith.constant 32 : index
      %c384 = arith.constant 384 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<128x384xf32>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<128xf32>
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %2 = gpu.thread_id  x
      %3 = affine.apply affine_map<()[s0, s1] -> (s1 * 2 + s0 floordiv 32)>()[%2, %workgroup_id_x]
      %4 = scf.for %arg0 = %c0 to %c384 step %c32 iter_args(%arg1 = %cst) -> (vector<1xf32>) {
        %6 = vector.transfer_read %0[%3, %arg0], %cst_0 {in_bounds = [true]} : memref<128x384xf32>, vector<32xf32>
        %7 = vector.broadcast %6 : vector<32xf32> to vector<1x32xf32>
        %8 = vector.multi_reduction <add>, %7, %arg1 [1] : vector<1x32xf32> to vector<1xf32>
        scf.yield %8 : vector<1xf32>
      }
      %5 = arith.divf %4, %cst_1 : vector<1xf32>
      vector.transfer_write %5, %1[%3] {in_bounds = [true]} : vector<1xf32>, memref<128xf32>
      return
    }
    }
  }
}

// CHECK-LABEL: func.func @simple_reduce() {
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : i32
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : i32
//   CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : i32
//   CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : i32
//   CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : i32
//   CHECK-DAG:   %[[C32:.*]] = arith.constant 32 : i32
//   CHECK-DAG:   %[[C32I:.*]] = arith.constant 32 : index
//   CHECK-DAG:   %[[TID:.*]] = gpu.thread_id  x
//   CHECK-DAG:   %[[VCST:.*]] = arith.constant dense<0.000000e+00> : vector<1xf32>
//       CHECK:   %[[F:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[V0:.*]] = %[[VCST]]) -> (vector<1xf32>) {
//   CHECK-DAG:     %[[E:.*]] = vector.extractelement %[[V0]][%[[C0]] : index] : vector<1xf32>
//   CHECK-DAG:     %[[ID:.*]] = affine.apply
//   CHECK-DAG:     %[[V1:.*]] = vector.transfer_read %{{.*}}[%{{.*}}, %[[ID]]], %{{.*}} {in_bounds = [true]} : memref<128x384xf32>, vector<1xf32>
//       CHECK:     %[[S:.*]] = vector.reduction <add>, %[[V1]] : vector<1xf32> into f32
//       CHECK:     %[[S0:.*]], %{{.*}} = gpu.shuffle  xor %[[S]], %[[C1]], %[[C32]] : f32
//       CHECK:     %[[S1:.*]] = arith.addf %[[S]], %[[S0]] : f32
//       CHECK:     %[[S2:.*]], %{{.*}} = gpu.shuffle  xor %[[S1]], %[[C2]], %[[C32]] : f32
//       CHECK:     %[[S3:.*]] = arith.addf %[[S1]], %[[S2]] : f32
//       CHECK:     %[[S4:.*]], %{{.*}} = gpu.shuffle  xor %[[S3]], %[[C4]], %[[C32]] : f32
//       CHECK:     %[[S5:.*]] = arith.addf %[[S3]], %[[S4]] : f32
//       CHECK:     %[[S6:.*]], %{{.*}} = gpu.shuffle  xor %[[S5]], %[[C8]], %[[C32]] : f32
//       CHECK:     %[[S7:.*]] = arith.addf %[[S5]], %[[S6]] : f32
//       CHECK:     %[[S8:.*]], %{{.*}} = gpu.shuffle  xor %[[S7]], %[[C16]], %[[C32]] : f32
//       CHECK:     %[[S9:.*]] = arith.addf %[[S7]], %[[S8]] : f32
//       CHECK:     %[[S10:.*]] = arith.addf %[[S9]], %[[E]] : f32
//       CHECK:     %[[B:.*]] = vector.broadcast %[[S10]] : f32 to vector<1xf32>
//       CHECK:     scf.yield %[[B]] : vector<1xf32>
//       CHECK:   }
//       CHECK:   %[[DIV:.*]] = arith.divf %[[F]], %{{.*}} : vector<1xf32>
//       CHECK:   %[[CMP:.*]] = arith.cmpi eq, %[[TID]], %[[C0]] : index
//       CHECK:   scf.if %[[CMP]] {
//       CHECK:     vector.transfer_write %[[DIV]], {{.*}} : vector<1xf32>, memref<128xf32>
//       CHECK:   }
//       CHECK:   return

// -----

// Make sure memref.load from uniform buffers are hoisted out as uniform code.

#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, uniform_buffer>
  ]>
]>
hal.executable private @reduce_uniform_buffer_offset  {
  hal.executable.variant @cuda target(#executable_target_cuda_nvptx_fb) {
    hal.executable.export @reduce_uniform_buffer_offset layout(#pipeline_layout) attributes {
      workgroup_size = [32 : index, 1 : index, 1 : index]
    }
    builtin.module {
    func.func @reduce_uniform_buffer_offset() {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cv = arith.constant dense<0.000000e+00> : vector<1xf32>
      %f0 = arith.constant 0.000000e+00 : f32
      %fv = arith.constant dense<3.840000e+02> : vector<1xf32>
      %c32 = arith.constant 32 : index
      %c384 = arith.constant 384 : index

      %ub = hal.interface.binding.subspan set(0) binding(2) type(uniform_buffer) offset(%c0) : memref<1xvector<4xi32>, #hal.descriptor_type<uniform_buffer>>
      %offsets = memref.load %ub[%c0] : memref<1xvector<4xi32>, #hal.descriptor_type<uniform_buffer>>
      %o0 = vector.extractelement %offsets[%c0 : index] : vector<4xi32>
      %o1 = vector.extractelement %offsets[%c1 : index] : vector<4xi32>
      %offset0 = arith.index_castui %o0 : i32 to index
      %offset1 = arith.index_castui %o1 : i32 to index

      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%offset0) : memref<128x384xf32>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%offset1) : memref<128xf32>
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %2 = gpu.thread_id  x
      %3 = affine.apply affine_map<()[s0, s1] -> (s1 * 2 + s0 floordiv 32)>()[%2, %workgroup_id_x]
      %4 = scf.for %arg0 = %c0 to %c384 step %c32 iter_args(%arg1 = %cv) -> (vector<1xf32>) {
        %6 = vector.transfer_read %0[%3, %arg0], %f0 {in_bounds = [true]} : memref<128x384xf32>, vector<32xf32>
        %7 = vector.broadcast %6 : vector<32xf32> to vector<1x32xf32>
        %8 = vector.multi_reduction <add>, %7, %arg1 [1] : vector<1x32xf32> to vector<1xf32>
        scf.yield %8 : vector<1xf32>
      }
      %5 = arith.divf %4, %fv : vector<1xf32>
      vector.transfer_write %5, %1[%3] {in_bounds = [true]} : vector<1xf32>, memref<128xf32>
      return
    }
    }
  }
}

//   CHECK-LABEL: func.func @reduce_uniform_buffer_offset()
//         CHECK:   %[[C0:.+]] = arith.constant 0 : index
//         CHECK:   %[[C1:.+]] = arith.constant 1 : index
//         CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(0) binding(2) type(uniform_buffer)
//         CHECK:   %[[LOAD:.+]] = memref.load %[[SUBSPAN]][%[[C0]]]
//         CHECK:   %[[EXT0:.+]] = vector.extractelement %[[LOAD]][%[[C0]] : index] : vector<4xi32>
//         CHECK:   %[[EXT1:.+]] = vector.extractelement %[[LOAD]][%[[C1]] : index] : vector<4xi32>
//         CHECK:   %[[OFFSET0:.+]] = arith.index_castui %[[EXT0]] : i32 to index
//         CHECK:   %[[OFFSET1:.+]] = arith.index_castui %[[EXT1]] : i32 to index
//         CHECK:   hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%[[OFFSET0]])
//         CHECK:   hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%[[OFFSET1]])
//         CHECK:   scf.for
//         CHECK:     vector.reduction
// CHECK-COUNT-5:     gpu.shuffle
//         CHECK:     scf.yield

// -----

// Make sure memref.load from readonly storage buffers are hoisted out as uniform code.

#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @reduce_storage_buffer_offset  {
  hal.executable.variant @cuda target(#executable_target_cuda_nvptx_fb) {
    hal.executable.export @reduce_storage_buffer_offset layout(#pipeline_layout) attributes {
      workgroup_size = [32 : index, 1 : index, 1 : index]
    }
    builtin.module {
    func.func @reduce_storage_buffer_offset() {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cv = arith.constant dense<0.000000e+00> : vector<1xf32>
      %f0 = arith.constant 0.000000e+00 : f32
      %fv = arith.constant dense<3.840000e+02> : vector<1xf32>
      %c32 = arith.constant 32 : index
      %c384 = arith.constant 384 : index

      %ub = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<1xvector<4xi32>, #hal.descriptor_type<storage_buffer>>
      %offsets = memref.load %ub[%c0] : memref<1xvector<4xi32>, #hal.descriptor_type<storage_buffer>>
      %o0 = vector.extractelement %offsets[%c0 : index] : vector<4xi32>
      %o1 = vector.extractelement %offsets[%c1 : index] : vector<4xi32>
      %offset0 = arith.index_castui %o0 : i32 to index
      %offset1 = arith.index_castui %o1 : i32 to index

      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%offset0) : memref<128x384xf32>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%offset1) : memref<128xf32>
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %2 = gpu.thread_id  x
      %3 = affine.apply affine_map<()[s0, s1] -> (s1 * 2 + s0 floordiv 32)>()[%2, %workgroup_id_x]
      %4 = scf.for %arg0 = %c0 to %c384 step %c32 iter_args(%arg1 = %cv) -> (vector<1xf32>) {
        %6 = vector.transfer_read %0[%3, %arg0], %f0 {in_bounds = [true]} : memref<128x384xf32>, vector<32xf32>
        %7 = vector.broadcast %6 : vector<32xf32> to vector<1x32xf32>
        %8 = vector.multi_reduction <add>, %7, %arg1 [1] : vector<1x32xf32> to vector<1xf32>
        scf.yield %8 : vector<1xf32>
      }
      %5 = arith.divf %4, %fv : vector<1xf32>
      vector.transfer_write %5, %1[%3] {in_bounds = [true]} : vector<1xf32>, memref<128xf32>
      return
    }
    }
  }
}

//   CHECK-LABEL: func.func @reduce_storage_buffer_offset()
//         CHECK:   %[[C0:.+]] = arith.constant 0 : index
//         CHECK:   %[[C1:.+]] = arith.constant 1 : index
//         CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
//         CHECK:   %[[LOAD:.+]] = memref.load %[[SUBSPAN]][%[[C0]]]
//         CHECK:   %[[EXT0:.+]] = vector.extractelement %[[LOAD]][%[[C0]] : index] : vector<4xi32>
//         CHECK:   %[[EXT1:.+]] = vector.extractelement %[[LOAD]][%[[C1]] : index] : vector<4xi32>
//         CHECK:   %[[OFFSET0:.+]] = arith.index_castui %[[EXT0]] : i32 to index
//         CHECK:   %[[OFFSET1:.+]] = arith.index_castui %[[EXT1]] : i32 to index
//         CHECK:   hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%[[OFFSET0]])
//         CHECK:   hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%[[OFFSET1]])
//         CHECK:   scf.for
//         CHECK:     vector.reduction
// CHECK-COUNT-5:     gpu.shuffle
//         CHECK:     scf.yield
