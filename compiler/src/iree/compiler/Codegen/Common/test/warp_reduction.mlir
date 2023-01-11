// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-reduction-to-gpu, cse)))))' %s | FileCheck %s

#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @simple_reduce  {
  hal.executable.variant @cuda, target = #executable_target_cuda_nvptx_fb {
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
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : memref<128x384xf32>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : memref<128xf32>
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
//       CHECK:     %[[ID:.*]] = affine.apply
//       CHECK:     %[[V1:.*]] = vector.transfer_read %{{.*}}[%{{.*}}, %[[ID]]], %{{.*}} {in_bounds = [true]} : memref<128x384xf32>, vector<1xf32>
//       CHECK:     %[[E:.*]] = vector.extract %[[V0]][0] : vector<1xf32>
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

#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @simple_half_reduce  {
  hal.executable.variant @cuda, target = #executable_target_cuda_nvptx_fb {
    hal.executable.export @simple_half_reduce layout(#pipeline_layout) attributes {
      workgroup_size = [32 : index, 1 : index, 1 : index]
    }
    builtin.module {
    func.func @simple_half_reduce() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant dense<0.000000e+00> : vector<1xf16>
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant dense<3.840000e+02> : vector<1xf16>
      %c32 = arith.constant 32 : index
      %c384 = arith.constant 384 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : memref<128x384xf16>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : memref<128xf16>
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %2 = gpu.thread_id  x
      %3 = affine.apply affine_map<()[s0, s1] -> (s1 * 2 + s0 floordiv 32)>()[%2, %workgroup_id_x]
      %4 = vector.transfer_read %0[%3, %c0], %cst_0 {in_bounds = [true]} : memref<128x384xf16>, vector<384xf16>
      %5 = vector.broadcast %4 : vector<384xf16> to vector<1x384xf16>
      %6 = vector.multi_reduction <add>, %5, %cst [1] : vector<1x384xf16> to vector<1xf16>
      %7 = arith.divf %6, %cst_1 : vector<1xf16>
      vector.transfer_write %7, %1[%3] {in_bounds = [true]} : vector<1xf16>, memref<128xf16>
      return
    }
    }
  }
}

// CHECK-LABEL: func.func @simple_half_reduce() {
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[IDENTITY:.*]] = arith.constant dense<0.000000e+00> : vector<2xf16>
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : i32
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : i32
//   CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : i32
//   CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : i32
//   CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : i32
//   CHECK-DAG:   %[[C32:.*]] = arith.constant 32 : i32
//   CHECK-DAG:   %[[C0F:.*]] = arith.constant 0.000000e+00 : f16
//   CHECK-DAG:   %[[TID:.*]] = gpu.thread_id  x
//       CHECK:   %[[READ_X:.*]] = affine.apply #{{.*}}()[%[[TID]], %{{.*}}]
//       CHECK:   %[[READ_Y:.*]] = affine.apply #{{.*}}()[%[[TID]]]
//       CHECK:   %[[V1:.*]] = vector.transfer_read %{{.*}}[%[[READ_X]], %[[READ_Y]]], %[[C0F]] {in_bounds = [true]} : memref<128x384xf16>, vector<12xf16>
//       CHECK:   %[[V2:.*]] = vector.extract_strided_slice %[[V1]] {offsets = [0], sizes = [6], strides = [1]} : vector<12xf16> to vector<6xf16>
//       CHECK:   %[[V3:.*]] = vector.reduction <add>, %[[V2]] : vector<6xf16> into f16
//       CHECK:   %[[V4:.*]] = vector.insert %[[V3]], %[[IDENTITY]] [0] : f16 into vector<2xf16>
//       CHECK:   %[[V5:.*]] = vector.extract_strided_slice %[[V1]] {offsets = [6], sizes = [6], strides = [1]} : vector<12xf16> to vector<6xf16>
//       CHECK:   %[[V6:.*]] = vector.reduction <add>, %[[V5]] : vector<6xf16> into f16
//       CHECK:   %[[V7:.*]] = vector.insert %[[V6]], %[[V4]] [1] : f16 into vector<2xf16>
//       CHECK:   %[[CAST0:.*]] = vector.bitcast %[[V7]] : vector<2xf16> to vector<1xi32>
//       CHECK:   %[[PACK0:.*]] = vector.extract %[[CAST0]][0] : vector<1xi32>
//       CHECK:   %[[S0:.*]], %{{.*}} = gpu.shuffle  xor %[[PACK0]], %[[C1]], %[[C32]] : i32
//       CHECK:   %[[BROADCAST0:.*]] = vector.broadcast %[[S0]] : i32 to vector<1xi32>
//       CHECK:   %[[UNPACK0:.*]] = vector.bitcast %[[BROADCAST0]] : vector<1xi32> to vector<2xf16>
//       CHECK:   %[[S1:.*]] = arith.addf %[[V7]], %[[UNPACK0]] : vector<2xf16>
//       CHECK:   %[[CAST1:.*]] = vector.bitcast %[[S1]] : vector<2xf16> to vector<1xi32>
//       CHECK:   %[[PACK1:.*]] = vector.extract %[[CAST1]][0] : vector<1xi32>
//       CHECK:   %[[S2:.*]], %{{.*}} = gpu.shuffle  xor %[[PACK1]], %[[C2]], %[[C32]] : i32
//       CHECK:   %[[BROADCAST1:.*]] = vector.broadcast %[[S2]] : i32 to vector<1xi32>
//       CHECK:   %[[UNPACK1:.*]] = vector.bitcast %[[BROADCAST1]] : vector<1xi32> to vector<2xf16>
//       CHECK:   %[[S3:.*]] = arith.addf %[[S1]], %[[UNPACK1]] : vector<2xf16>
//       CHECK:   %[[CAST2:.*]] = vector.bitcast %[[S3]] : vector<2xf16> to vector<1xi32>
//       CHECK:   %[[PACK2:.*]] = vector.extract %[[CAST2]][0] : vector<1xi32>
//       CHECK:   %[[S4:.*]], %{{.*}} = gpu.shuffle  xor %[[PACK2]], %[[C4]], %[[C32]] : i32
//       CHECK:   %[[BROADCAST2:.*]] = vector.broadcast %[[S4]] : i32 to vector<1xi32>
//       CHECK:   %[[UNPACK2:.*]] = vector.bitcast %[[BROADCAST2]] : vector<1xi32> to vector<2xf16>
//       CHECK:   %[[S5:.*]] = arith.addf %[[S3]], %[[UNPACK2]] : vector<2xf16>
//       CHECK:   %[[CAST3:.*]] = vector.bitcast %[[S5]] : vector<2xf16> to vector<1xi32>
//       CHECK:   %[[PACK3:.*]] = vector.extract %[[CAST3]][0] : vector<1xi32>
//       CHECK:   %[[S6:.*]], %{{.*}} = gpu.shuffle  xor %[[PACK3]], %[[C8]], %[[C32]] : i32
//       CHECK:   %[[BROADCAST3:.*]] = vector.broadcast %[[S6]] : i32 to vector<1xi32>
//       CHECK:   %[[UNPACK3:.*]] = vector.bitcast %[[BROADCAST3]] : vector<1xi32> to vector<2xf16>
//       CHECK:   %[[S7:.*]] = arith.addf %[[S5]], %[[UNPACK3]] : vector<2xf16>
//       CHECK:   %[[CAST4:.*]] = vector.bitcast %[[S7]] : vector<2xf16> to vector<1xi32>
//       CHECK:   %[[PACK4:.*]] = vector.extract %[[CAST4]][0] : vector<1xi32>
//       CHECK:   %[[S8:.*]], %{{.*}} = gpu.shuffle  xor %[[PACK4]], %[[C16]], %[[C32]] : i32
//       CHECK:   %[[BROADCAST4:.*]] = vector.broadcast %[[S8]] : i32 to vector<1xi32>
//       CHECK:   %[[UNPACK4:.*]] = vector.bitcast %[[BROADCAST4]] : vector<1xi32> to vector<2xf16>
//       CHECK:   %[[S9:.*]] = arith.addf %[[S7]], %[[UNPACK4]] : vector<2xf16>
//       CHECK:   %[[S10:.*]] = vector.reduction <add>, %[[S9]] : vector<2xf16> into f16
//       CHECK:   %[[S11:.*]] = arith.addf %[[S10]], %[[C0F]] : f16
//       CHECK:   %[[B:.*]] = vector.broadcast %[[S11]] : f16 to vector<1xf16>
//       CHECK:   %[[DIV:.*]] = arith.divf %[[B]], %{{.*}} : vector<1xf16>
//       CHECK:   %[[CMP:.*]] = arith.cmpi eq, %[[TID]], %[[C0]] : index
//       CHECK:   scf.if %[[CMP]] {
//       CHECK:     vector.transfer_write %[[DIV]], {{.*}} : vector<1xf16>, memref<128xf16>
//       CHECK:   }
//       CHECK:   return

// -----

#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @simple_half_reduce_small  {
  hal.executable.variant @cuda, target = #executable_target_cuda_nvptx_fb {
    hal.executable.export @simple_half_reduce_small layout(#pipeline_layout) attributes {
      workgroup_size = [32 : index, 1 : index, 1 : index]
    }
    builtin.module {
    func.func @simple_half_reduce_small() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant dense<0.000000e+00> : vector<1xf16>
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant dense<3.840000e+02> : vector<1xf16>
      %c32 = arith.constant 32 : index
      %c384 = arith.constant 384 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : memref<128x32xf16>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : memref<128xf16>
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %2 = gpu.thread_id  x
      %3 = affine.apply affine_map<()[s0, s1] -> (s1 * 2 + s0 floordiv 32)>()[%2, %workgroup_id_x]
      %4 = vector.transfer_read %0[%3, %c0], %cst_0 {in_bounds = [true]} : memref<128x32xf16>, vector<32xf16>
      %5 = vector.broadcast %4 : vector<32xf16> to vector<1x32xf16>
      %6 = vector.multi_reduction <add>, %5, %cst [1] : vector<1x32xf16> to vector<1xf16>
      %7 = arith.divf %6, %cst_1 : vector<1xf16>
      vector.transfer_write %7, %1[%3] {in_bounds = [true]} : vector<1xf16>, memref<128xf16>
      return
    }
    }
  }
}

// CHECK-LABEL: func.func @simple_half_reduce_small() {
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[IDENTITY:.*]] = arith.constant dense<0.000000e+00> : vector<2xf16>
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : i32
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : i32
//   CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : i32
//   CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : i32
//   CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : i32
//   CHECK-DAG:   %[[C32:.*]] = arith.constant 32 : i32
//   CHECK-DAG:   %[[C1F:.*]] = arith.constant 0.000000e+00 : f16
//   CHECK-DAG:   %[[TID:.*]] = gpu.thread_id  x
//       CHECK:   %[[ID:.*]] = affine.apply
//       CHECK:   %[[V1:.*]] = vector.transfer_read %{{.*}}[%[[ID]], %{{.*}}], %{{.*}} {in_bounds = [true]} : memref<128x32xf16>, vector<1xf16>
//       CHECK:   %[[V2:.*]] = vector.insert_strided_slice %[[V1]], %[[IDENTITY]] {offsets = [0], strides = [1]} : vector<1xf16> into vector<2xf16>
//       CHECK:   %[[CAST0:.*]] = vector.bitcast %[[V2]] : vector<2xf16> to vector<1xi32>
//       CHECK:   %[[PACK0:.*]] = vector.extract %[[CAST0]][0] : vector<1xi32>
//       CHECK:   %[[S0:.*]], %{{.*}} = gpu.shuffle  xor %[[PACK0]], %[[C1]], %[[C32]] : i32
//       CHECK:   %[[BROADCAST0:.*]] = vector.broadcast %[[S0]] : i32 to vector<1xi32>
//       CHECK:   %[[UNPACK0:.*]] = vector.bitcast %[[BROADCAST0]] : vector<1xi32> to vector<2xf16>
//       CHECK:   %[[S1:.*]] = arith.addf %[[V2]], %[[UNPACK0]] : vector<2xf16>
//       CHECK:   %[[CAST1:.*]] = vector.bitcast %[[S1]] : vector<2xf16> to vector<1xi32>
//       CHECK:   %[[PACK1:.*]] = vector.extract %[[CAST1]][0] : vector<1xi32>
//       CHECK:   %[[S2:.*]], %{{.*}} = gpu.shuffle  xor %[[PACK1]], %[[C2]], %[[C32]] : i32
//       CHECK:   %[[BROADCAST1:.*]] = vector.broadcast %[[S2]] : i32 to vector<1xi32>
//       CHECK:   %[[UNPACK1:.*]] = vector.bitcast %[[BROADCAST1]] : vector<1xi32> to vector<2xf16>
//       CHECK:   %[[S3:.*]] = arith.addf %[[S1]], %[[UNPACK1]] : vector<2xf16>
//       CHECK:   %[[CAST2:.*]] = vector.bitcast %[[S3]] : vector<2xf16> to vector<1xi32>
//       CHECK:   %[[PACK2:.*]] = vector.extract %[[CAST2]][0] : vector<1xi32>
//       CHECK:   %[[S4:.*]], %{{.*}} = gpu.shuffle  xor %[[PACK2]], %[[C4]], %[[C32]] : i32
//       CHECK:   %[[BROADCAST2:.*]] = vector.broadcast %[[S4]] : i32 to vector<1xi32>
//       CHECK:   %[[UNPACK2:.*]] = vector.bitcast %[[BROADCAST2]] : vector<1xi32> to vector<2xf16>
//       CHECK:   %[[S5:.*]] = arith.addf %[[S3]], %[[UNPACK2]] : vector<2xf16>
//       CHECK:   %[[CAST3:.*]] = vector.bitcast %[[S5]] : vector<2xf16> to vector<1xi32>
//       CHECK:   %[[PACK3:.*]] = vector.extract %[[CAST3]][0] : vector<1xi32>
//       CHECK:   %[[S6:.*]], %{{.*}} = gpu.shuffle  xor %[[PACK3]], %[[C8]], %[[C32]] : i32
//       CHECK:   %[[BROADCAST3:.*]] = vector.broadcast %[[S6]] : i32 to vector<1xi32>
//       CHECK:   %[[UNPACK3:.*]] = vector.bitcast %[[BROADCAST3]] : vector<1xi32> to vector<2xf16>
//       CHECK:   %[[S7:.*]] = arith.addf %[[S5]], %[[UNPACK3]] : vector<2xf16>
//       CHECK:   %[[CAST4:.*]] = vector.bitcast %[[S7]] : vector<2xf16> to vector<1xi32>
//       CHECK:   %[[PACK4:.*]] = vector.extract %[[CAST4]][0] : vector<1xi32>
//       CHECK:   %[[S8:.*]], %{{.*}} = gpu.shuffle  xor %[[PACK4]], %[[C16]], %[[C32]] : i32
//       CHECK:   %[[BROADCAST4:.*]] = vector.broadcast %[[S8]] : i32 to vector<1xi32>
//       CHECK:   %[[UNPACK4:.*]] = vector.bitcast %[[BROADCAST4]] : vector<1xi32> to vector<2xf16>
//       CHECK:   %[[S9:.*]] = arith.addf %[[S7]], %[[UNPACK4]] : vector<2xf16>
//       CHECK:   %[[S10:.*]] = vector.reduction <add>, %[[S9]] : vector<2xf16> into f16
//       CHECK:   %[[S11:.*]] = arith.addf %[[S10]], %[[C1F]] : f16
//       CHECK:   %[[B:.*]] = vector.broadcast %[[S11]] : f16 to vector<1xf16>
//       CHECK:   %[[DIV:.*]] = arith.divf %[[B]], %{{.*}} : vector<1xf16>
//       CHECK:   %[[CMP:.*]] = arith.cmpi eq, %[[TID]], %[[C0]] : index
//       CHECK:   scf.if %[[CMP]] {
//       CHECK:     vector.transfer_write %[[DIV]], {{.*}} : vector<1xf16>, memref<128xf16>
//       CHECK:   }
//       CHECK:   return

// -----

#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @simple_quarter_reduce  {
  hal.executable.variant @cuda, target = #executable_target_cuda_nvptx_fb {
    hal.executable.export @simple_quarter_reduce layout(#pipeline_layout) attributes {
      workgroup_size = [32 : index, 1 : index, 1 : index]
    }
    builtin.module {
    func.func @simple_quarter_reduce() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant dense<0> : vector<1xi8>
      %cst_0 = arith.constant 0 : i8
      %c32 = arith.constant 32 : index
      %c384 = arith.constant 384 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : memref<128x384xi8>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : memref<128xi8>
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %2 = gpu.thread_id  x
      %3 = affine.apply affine_map<()[s0, s1] -> (s1 * 2 + s0 floordiv 32)>()[%2, %workgroup_id_x]
      %4 = vector.transfer_read %0[%3, %c0], %cst_0 {in_bounds = [true]} : memref<128x384xi8>, vector<32xi8>
      %5 = vector.broadcast %4 : vector<32xi8> to vector<1x32xi8>
      %6 = vector.multi_reduction <add>, %5, %cst [1] : vector<1x32xi8> to vector<1xi8>
      vector.transfer_write %6, %1[%3] {in_bounds = [true]} : vector<1xi8>, memref<128xi8>
      return
    }
    }
  }
}

// CHECK-LABEL: func.func @simple_quarter_reduce() {
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[IDENTITY:.*]] = arith.constant dense<0> : vector<4xi8>
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : i32
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : i32
//   CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : i32
//   CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : i32
//   CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : i32
//   CHECK-DAG:   %[[C32:.*]] = arith.constant 32 : i32
//   CHECK-DAG:   %[[TID:.*]] = gpu.thread_id  x
//       CHECK:   %[[ID:.*]] = affine.apply
//       CHECK:   %[[V1:.*]] = vector.transfer_read %{{.*}}[%[[ID]], %{{.*}}], %{{.*}} {in_bounds = [true]} : memref<128x384xi8>, vector<1xi8>
//       CHECK:   %[[V2:.*]] = vector.insert_strided_slice %[[V1]], %[[IDENTITY]] {offsets = [0], strides = [1]} : vector<1xi8> into vector<4xi8>
//       CHECK:   %[[CAST0:.*]] = vector.bitcast %[[V2]] : vector<4xi8> to vector<1xi32>
//       CHECK:   %[[PACK0:.*]] = vector.extract %[[CAST0]][0] : vector<1xi32>
//       CHECK:   %[[S0:.*]], %{{.*}} = gpu.shuffle  xor %[[PACK0]], %[[C1]], %[[C32]] : i32
//       CHECK:   %[[BROADCAST0:.*]] = vector.broadcast %[[S0]] : i32 to vector<1xi32>
//       CHECK:   %[[UNPACK0:.*]] = vector.bitcast %[[BROADCAST0]] : vector<1xi32> to vector<4xi8>
//       CHECK:   %[[S1:.*]] = arith.addi %[[V2]], %[[UNPACK0]] : vector<4xi8>
//       CHECK:   %[[CAST1:.*]] = vector.bitcast %[[S1]] : vector<4xi8> to vector<1xi32>
//       CHECK:   %[[PACK1:.*]] = vector.extract %[[CAST1]][0] : vector<1xi32>
//       CHECK:   %[[S2:.*]], %{{.*}} = gpu.shuffle  xor %[[PACK1]], %[[C2]], %[[C32]] : i32
//       CHECK:   %[[BROADCAST1:.*]] = vector.broadcast %[[S2]] : i32 to vector<1xi32>
//       CHECK:   %[[UNPACK1:.*]] = vector.bitcast %[[BROADCAST1]] : vector<1xi32> to vector<4xi8>
//       CHECK:   %[[S3:.*]] = arith.addi %[[S1]], %[[UNPACK1]] : vector<4xi8>
//       CHECK:   %[[CAST2:.*]] = vector.bitcast %[[S3]] : vector<4xi8> to vector<1xi32>
//       CHECK:   %[[PACK2:.*]] = vector.extract %[[CAST2]][0] : vector<1xi32>
//       CHECK:   %[[S4:.*]], %{{.*}} = gpu.shuffle  xor %[[PACK2]], %[[C4]], %[[C32]] : i32
//       CHECK:   %[[BROADCAST2:.*]] = vector.broadcast %[[S4]] : i32 to vector<1xi32>
//       CHECK:   %[[UNPACK2:.*]] = vector.bitcast %[[BROADCAST2]] : vector<1xi32> to vector<4xi8>
//       CHECK:   %[[S5:.*]] = arith.addi %[[S3]], %[[UNPACK2]] : vector<4xi8>
//       CHECK:   %[[CAST3:.*]] = vector.bitcast %[[S5]] : vector<4xi8> to vector<1xi32>
//       CHECK:   %[[PACK3:.*]] = vector.extract %[[CAST3]][0] : vector<1xi32>
//       CHECK:   %[[S6:.*]], %{{.*}} = gpu.shuffle  xor %[[PACK3]], %[[C8]], %[[C32]] : i32
//       CHECK:   %[[BROADCAST3:.*]] = vector.broadcast %[[S6]] : i32 to vector<1xi32>
//       CHECK:   %[[UNPACK3:.*]] = vector.bitcast %[[BROADCAST3]] : vector<1xi32> to vector<4xi8>
//       CHECK:   %[[S7:.*]] = arith.addi %[[S5]], %[[UNPACK3]] : vector<4xi8>
//       CHECK:   %[[CAST4:.*]] = vector.bitcast %[[S7]] : vector<4xi8> to vector<1xi32>
//       CHECK:   %[[PACK4:.*]] = vector.extract %[[CAST4]][0] : vector<1xi32>
//       CHECK:   %[[S8:.*]], %{{.*}} = gpu.shuffle  xor %[[PACK4]], %[[C16]], %[[C32]] : i32
//       CHECK:   %[[BROADCAST4:.*]] = vector.broadcast %[[S8]] : i32 to vector<1xi32>
//       CHECK:   %[[UNPACK4:.*]] = vector.bitcast %[[BROADCAST4]] : vector<1xi32> to vector<4xi8>
//       CHECK:   %[[S9:.*]] = arith.addi %[[S7]], %[[UNPACK4]] : vector<4xi8>
//       CHECK:   %[[S10:.*]] = vector.reduction <add>, %[[S9]] : vector<4xi8> into i8
//       CHECK:   %[[B:.*]] = vector.broadcast %[[S10]] : i8 to vector<1xi8>
//       CHECK:   %[[CMP:.*]] = arith.cmpi eq, %[[TID]], %[[C0]] : index
//       CHECK:   scf.if %[[CMP]] {
//       CHECK:     vector.transfer_write %[[B]], {{.*}} : vector<1xi8>, memref<128xi8>
//       CHECK:   }
//       CHECK:   return

// -----

#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @simple_reduce_multi_warp  {
  hal.executable.variant @cuda, target = #executable_target_cuda_nvptx_fb {
    hal.executable.export @simple_reduce_multi_warp layout(#pipeline_layout) attributes {
      workgroup_size = [64 : index, 1 : index, 1 : index]
    }
    builtin.module {
    func.func @simple_reduce_multi_warp() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
      %cst_0 = arith.constant 0.000000e+00 : f32
      %cst_1 = arith.constant dense<3.840000e+02> : vector<1xf32>
      %c64 = arith.constant 64 : index
      %c384 = arith.constant 384 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : memref<128x384xf32>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : memref<128xf32>
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %2 = gpu.thread_id  x
      %3 = affine.apply affine_map<()[s0, s1] -> (s1 * 2 + s0 floordiv 32)>()[%2, %workgroup_id_x]
      %4 = scf.for %arg0 = %c0 to %c384 step %c64 iter_args(%arg1 = %cst) -> (vector<1xf32>) {
        %6 = vector.transfer_read %0[%3, %arg0], %cst_0 {in_bounds = [true]} : memref<128x384xf32>, vector<64xf32>
        %7 = vector.broadcast %6 : vector<64xf32> to vector<1x64xf32>
        %8 = vector.multi_reduction <add>, %7, %arg1 [1] : vector<1x64xf32> to vector<1xf32>
        scf.yield %8 : vector<1xf32>
      }
      %5 = arith.divf %4, %cst_1 : vector<1xf32>
      vector.transfer_write %5, %1[%3] {in_bounds = [true]} : vector<1xf32>, memref<128xf32>
      return
    }
    }
  }
}

// CHECK-LABEL: func.func @simple_reduce_multi_warp() {
//   CHECK-DAG:   %[[C0I:.*]] = arith.constant 0 : i32
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : i32
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : i32
//   CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : i32
//   CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : i32
//   CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : i32
//   CHECK-DAG:   %[[C32:.*]] = arith.constant 32 : i32
//   CHECK-DAG:   %[[C32I:.*]] = arith.constant 32 : index
//   CHECK-DAG:   %[[IDENTITY:.*]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[TID:.*]] = gpu.thread_id  x
//   CHECK-DAG:   %[[VCST:.*]] = arith.constant dense<0.000000e+00> : vector<1xf32>
//       CHECK:   %[[F:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[V0:.*]] = %[[VCST]]) -> (vector<1xf32>) {
//       CHECK:     %[[ID:.*]] = affine.apply
//       CHECK:     %[[V1:.*]] = vector.transfer_read %{{.*}}[%{{.*}}, %[[ID]]], %{{.*}} {in_bounds = [true]} : memref<128x384xf32>, vector<1xf32>
//       CHECK:     %[[E:.*]] = vector.extract %[[V0]][0] : vector<1xf32>
//       CHECK:     %[[S:.*]] = vector.reduction <add>, %[[V1]] : vector<1xf32>
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
//       CHECK:     %[[A:.*]] = memref.alloc() : memref<2xf32, 3>
//       CHECK:     %[[WID:.*]] = arith.divui %[[TID]], %[[C32I]] : index
//       CHECK:     %[[LANE_ID:.*]] = arith.remui %[[TID]], %[[C32I]] : index
//       CHECK:     %[[LANE0:.*]] = arith.cmpi eq, %[[LANE_ID]], %[[C0]] : index
//       CHECK:     scf.if %[[LANE0]] { 
//       CHECK:       memref.store %[[S9]], %[[A]][%[[WID]]] : memref<2xf32, 3>
//       CHECK:     }
//       CHECK:     gpu.barrier
//       CHECK:     %[[LOAD_VAL:.*]] = memref.load %[[A]][%[[LANE_ID]]] : memref<2xf32, 3>
//       CHECK:     %[[S10:.*]], %{{.*}} = gpu.shuffle  xor %[[LOAD_VAL]], %[[C1]], %[[C32]] : f32
//       CHECK:     %[[S11:.*]] = arith.addf %[[LOAD_VAL]], %[[S10]] : f32
//       CHECK:     %[[S12:.*]], %{{.*}} = gpu.shuffle  idx %[[S11]], %[[C0I]], %[[C32]] : f32
//       CHECK:     %[[S20:.*]] = arith.addf %[[S12]], %[[E]] : f32
//       CHECK:     %[[B:.*]] = vector.broadcast %[[S20]] : f32 to vector<1xf32>
//       CHECK:     scf.yield %[[B]] : vector<1xf32>
//       CHECK:   }
//       CHECK:   %[[DIV:.*]] = arith.divf %[[F]], %{{.*}} : vector<1xf32>
//       CHECK:   %[[CMP:.*]] = arith.cmpi eq, %[[TID]], %[[C0]] : index
//       CHECK:   scf.if %[[CMP]] {
//       CHECK:     vector.transfer_write %[[DIV]], {{.*}} : vector<1xf32>, memref<128xf32>
//       CHECK:   }
//       CHECK:   return

// -----

#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @reduce_then_broadcast  {
  hal.executable.variant @cuda, target = #executable_target_cuda_nvptx_fb {
    hal.executable.export @reduce_then_broadcast layout(#pipeline_layout) attributes {
      workgroup_size = [64 : index, 1 : index, 1 : index]
    }
    builtin.module {
    func.func @reduce_then_broadcast() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant dense<0.000000e+00> : vector<1x1xf32>
      %cst_0 = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c384 = arith.constant 384 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : memref<12x512x512xf32>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : memref<12x512x512xf32>
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %workgroup_id_y = hal.interface.workgroup.id[1] : index
      %6 = vector.transfer_read %0[%workgroup_id_y, %workgroup_id_x, %c0], %cst_0 {in_bounds = [true]} : memref<12x512x512xf32>, vector<512xf32>
      %7 = vector.broadcast %6 : vector<512xf32> to vector<1x1x512xf32>
      %8 = vector.multi_reduction <maxf>, %7, %cst [2] : vector<1x1x512xf32> to vector<1x1xf32>
      %9 = vector.broadcast %8 : vector<1x1xf32> to vector<1x1x512xf32>
      %10 = vector.extract %9[0, 0] : vector<1x1x512xf32>
      %11 = arith.subf %6, %10 : vector<512xf32>
      %12 = math.exp %11 : vector<512xf32>
      vector.transfer_write %12, %1[%workgroup_id_y, %workgroup_id_x, %c0] {in_bounds = [true]} : vector<512xf32>, memref<12x512x512xf32>
      return
    }
    }
  }
}

// Check that there is no scf.if generated.
// If some operations were not distributed we would end up with a scf.if(warp0) block.
// CHECK-LABEL: func.func @reduce_then_broadcast() {
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C32I:.*]] = arith.constant 32 : index
//   CHECK-DAG:   %[[TID:.*]] = gpu.thread_id  x
//       CHECK:   %[[LANE_ID:.*]] = arith.remui %[[TID]], %[[C32I]] : index
//       CHECK:   %[[LANE0:.*]] = arith.cmpi eq, %[[LANE_ID]], %[[C0]] : index
//       CHECK:   scf.if %[[LANE0]] { 
//       CHECK:     memref.store {{.*}} : memref<2xf32, 3>
//       CHECK:   }
//   CHECK-NOT:  scf.if
//       CHECK:  return

// -----

#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @reduce_odd_num_warp  {
  hal.executable.variant @cuda, target = #executable_target_cuda_nvptx_fb {
    hal.executable.export @reduce_odd_num_warp layout(#pipeline_layout) attributes {
      workgroup_size = [96 : index, 1 : index, 1 : index]
    }
    builtin.module {
    func.func @reduce_odd_num_warp() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
      %cst_0 = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : memref<768xf32>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : memref<1xf32>
      %6 = vector.transfer_read %0[%c0], %cst_0 {in_bounds = [true]} : memref<768xf32>, vector<768xf32>
      %7 = vector.broadcast %6 : vector<768xf32> to vector<1x768xf32>
      %8 = vector.multi_reduction <maxf>, %7, %cst [1] : vector<1x768xf32> to vector<1xf32>
      vector.transfer_write %8, %1[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32>
      return
    }
    }
  }
}

// CHECK-LABEL: func.func @reduce_odd_num_warp() {
//   CHECK-DAG:   %[[C0I:.+]] = arith.constant 0 : i32
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : i32
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : i32
//   CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C32I:.+]] = arith.constant 32 : i32
//   CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
//   CHECK-DAG:   %[[CF:.+]] = arith.constant 0xFF800000 : f32
//       CHECK:   %[[ID:.+]] = gpu.thread_id  x
//       CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<3xf32, 3>
//       CHECK:   %[[WARP_ID:.+]] = arith.divui %[[ID]], %[[C32]] : index
//       CHECK:   %[[LANE_ID:.+]] = arith.remui %[[ID]], %[[C32]] : index

//       CHECK:   gpu.barrier
//       CHECK:   %[[V:.+]] = memref.load %[[ALLOC]][%[[LANE_ID]]] : memref<3xf32, 3>
//       CHECK:   %[[C:.+]] = arith.cmpi sge, %[[LANE_ID]], %[[C3]] : index
//       CHECK:   %[[SEL:.+]] = arith.select %[[C]], %[[CF]], %[[V]] : f32
//       CHECK:   %[[S0:.+]], %{{.*}} = gpu.shuffle  xor %[[SEL]], %[[C1]], %[[C32I]] : f32
//       CHECK:   %[[S1:.+]] = arith.maxf %[[SEL]], %[[S0]] : f32
//       CHECK:   %[[S2:.+]], %{{.*}} = gpu.shuffle  xor %[[S1]], %[[C2]], %[[C32I]] : f32
//       CHECK:   %[[S3:.+]] = arith.maxf %[[S1]], %[[S2]] : f32
//       CHECK:   %[[S4:.+]], %{{.*}} = gpu.shuffle  idx %[[S3]], %[[C0I]], %[[C32I]] : f32
//       CHECK:   %[[S5:.+]] = arith.maxf %[[S4]], %cst_0 : f32
//       CHECK:   %[[B:.+]] = vector.broadcast %[[S5]] : f32 to vector<1xf32>
//       CHECK:   %[[LANE0:.+]] = arith.cmpi eq, %[[ID]], %[[C0]] : index
//       CHECK:   scf.if %[[LANE0]] {
//       CHECK:     vector.transfer_write %[[B]], %{{.*}}[%[[C0]]] {in_bounds = [true]} : vector<1xf32>, memref<1xf32>
//       CHECK:   }
//       CHECK:   return
