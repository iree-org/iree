// RUN: iree-opt -pass-pipeline="hal.executable(hal.executable.target(iree-codegen-hlo-to-nvvm-pipeline))" %s | IreeFileCheck %s

// Verify that a simple element wise op gets lowered succefully all the way to
// nvvm/llvm dialect.

hal.executable @simpleMath_ex_dispatch_0 {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @cuda, filter="cuda" {
  hal.executable.entry_point @add_dispatch_0 attributes {interface = @legacy_io, ordinal = 0 : index, signature = (!flow.dispatch.tensor<readonly:16xf32>, !flow.dispatch.tensor<readonly:16xf32>, !flow.dispatch.tensor<writeonly:16xf32>) -> ()}
  module  {
    func @add_dispatch_0() {
      %c0 = constant 0 : index
      %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : !flow.dispatch.tensor<readonly:16xf32>
      %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : !flow.dispatch.tensor<readonly:16xf32>
      %2 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:16xf32>
      %3 = linalg.init_tensor [16] : tensor<16xf32>
      %4 = flow.dispatch.tensor.load %0 : !flow.dispatch.tensor<readonly:16xf32> -> tensor<16xf32>
      %5 = flow.dispatch.tensor.load %1 : !flow.dispatch.tensor<readonly:16xf32> -> tensor<16xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4, %5 : tensor<16xf32>, tensor<16xf32>) outs(%3 : tensor<16xf32>) {
      ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
          %7 = addf %arg0, %arg1 : f32
          linalg.yield %7 : f32
        } -> tensor<16xf32>
        flow.dispatch.tensor.store %6, %2 : tensor<16xf32> -> !flow.dispatch.tensor<writeonly:16xf32>
        return
      }
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

// CHECK-LABEL: hal.executable @simpleMath_ex_dispatch_0
//       CHECK:   hal.executable.target @cuda, filter="cuda" {
//       CHECK:   llvm.fadd
