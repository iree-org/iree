// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(iree-codegen-linalg-to-rocdl-pipeline))' %s | IreeFileCheck %s

// Verify that a simple element wise op gets lowered succefully all the way to
// nvvm/llvm dialect.

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @simpleMath_ex_dispatch_0 {
  hal.executable.variant @rocm, target = <"rocm", "rocm-hsaco-fb"> {
  hal.executable.entry_point @add_dispatch_0 layout(#executable_layout)
  builtin.module {
    func @add_dispatch_0() {
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(0) : !flow.dispatch.tensor<readonly:16xf32>
      %1 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(1) : !flow.dispatch.tensor<readonly:16xf32>
      %2 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(2) : !flow.dispatch.tensor<writeonly:16xf32>
      %3 = linalg.init_tensor [16] : tensor<16xf32>
      %4 = flow.dispatch.tensor.load %0, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:16xf32> -> tensor<16xf32>
      %5 = flow.dispatch.tensor.load %1, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:16xf32> -> tensor<16xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4, %5 : tensor<16xf32>, tensor<16xf32>) outs(%3 : tensor<16xf32>) {
      ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
          %7 = arith.addf %arg0, %arg1 : f32
          linalg.yield %7 : f32
        } -> tensor<16xf32>
        flow.dispatch.tensor.store %6, %2, offsets=[], sizes=[], strides=[] : tensor<16xf32> -> !flow.dispatch.tensor<writeonly:16xf32>
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable public @simpleMath_ex_dispatch_0
//       CHECK:   hal.executable.variant public @rocm
//       CHECK:   llvm.fadd

// -----

#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0] -> (s0, -d0 + 1024)>
#map2 = affine_map<(d0)[s0] -> (-d0 + 1024, s0)>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @dot_dispatch_0 {
  hal.executable.variant @rocm, target = <"rocm", "rocm-hsaco-fb"> {
    hal.executable.entry_point @dot_dispatch_0 layout(#executable_layout)
    builtin.module {
      func @dot_dispatch_0() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c1 = arith.constant 1 : index
        %0 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(0) : !flow.dispatch.tensor<readonly:1024x1024xf32>
        %1 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(1) : !flow.dispatch.tensor<readonly:1024x1024xf32>
        %2 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(2) : !flow.dispatch.tensor<writeonly:1024x1024xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %3 = affine.apply #map0()[%workgroup_id_y, %workgroup_size_y]
        %4 = affine.apply #map0()[%workgroup_count_y, %workgroup_size_y]
        scf.for %arg0 = %3 to %c1024 step %4 {
          %5 = affine.apply #map0()[%workgroup_id_x, %workgroup_size_x]
          %6 = affine.apply #map0()[%workgroup_count_x, %workgroup_size_x]
          scf.for %arg1 = %5 to %c1024 step %6 {
            %7 = affine.min #map1(%arg0)[%workgroup_size_y]
            %8 = flow.dispatch.tensor.load %0, offsets = [%arg0, %c0], sizes = [%7, %c1024], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:1024x1024xf32> -> tensor<?x1024xf32>
            %9 = affine.min #map1(%arg1)[%workgroup_size_x]
            %10 = flow.dispatch.tensor.load %1, offsets = [%c0, %arg1], sizes = [%c1024, %9], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:1024x1024xf32> -> tensor<1024x?xf32>
            %11 = affine.min #map1(%arg0)[%workgroup_size_y]
            %12 = affine.min #map1(%arg1)[%workgroup_size_x]
            %13 = affine.min #map2(%arg0)[%workgroup_size_y]
            %14 = affine.min #map2(%arg1)[%workgroup_size_x]
            %15 = linalg.init_tensor [%13, %14] : tensor<?x?xf32>
            %16 = linalg.fill(%cst, %15) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
            %17 = linalg.matmul ins(%8, %10 : tensor<?x1024xf32>, tensor<1024x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
            flow.dispatch.tensor.store %17, %2, offsets = [%arg0, %arg1], sizes = [%11, %12], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:1024x1024xf32>
          }
        }
        return
      }
    }
  }
}

//   CHECK-LABEL: hal.executable public @dot_dispatch_0
//         CHECK:   hal.executable.variant public @rocm
//       CHECK-NOT:   llvm.store
//   CHECK-COUNT-3:   llvm.load {{.*}} : !llvm.ptr<vector<4xf32>>
//           CHECK:   llvm.br
//   CHECK-COUNT-3:   llvm.store {{.*}} : !llvm.ptr<vector<4xf32>, 3>
//  CHECK-COUNT-32:   llvm.load {{.*}} : !llvm.ptr<vector<4xf32>, 3>
// CHECK-COUNT-128:   "llvm.intr.fmuladd"({{.*}}) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
//   CHECK-COUNT-3:   llvm.load {{.*}} : !llvm.ptr<vector<4xf32>>
//           CHECK:   llvm.br
//   CHECK-COUNT-3:   llvm.store {{.*}} : !llvm.ptr<vector<4xf32>, 3>
//  CHECK-COUNT-32:   llvm.load {{.*}} : !llvm.ptr<vector<4xf32>, 3>
// CHECK-COUNT-128:   "llvm.intr.fmuladd"({{.*}}) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
//   CHECK-COUNT-4:   llvm.store {{.*}} : !llvm.ptr<vector<4xf32>>
