// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx908 --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-codegen-llvmgpu-configuration-pipeline), iree-codegen-linalg-to-rocdl-pipeline)))" %s | FileCheck %s --check-prefix=CDNA1
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-codegen-llvmgpu-configuration-pipeline), iree-codegen-linalg-to-rocdl-pipeline)))" %s | FileCheck %s --check-prefix=CDNA3
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1100 --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-codegen-llvmgpu-configuration-pipeline), iree-codegen-linalg-to-rocdl-pipeline)))" %s | FileCheck %s --check-prefix=RDNA3

// Verify that a simple element wise op gets lowered successfully all the way to
// nvvm/llvm dialect.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @simpleMath_ex_dispatch_0 {
  hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export @add_dispatch_0 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @add_dispatch_0() {
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<16xf32>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<16xf32>>
      %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<16xf32>>
      %3 = tensor.empty() : tensor<16xf32>
      %4 = flow.dispatch.tensor.load %0, offsets=[0], sizes=[16], strides=[1] : !flow.dispatch.tensor<readonly:tensor<16xf32>> -> tensor<16xf32>
      %5 = flow.dispatch.tensor.load %1, offsets=[0], sizes=[16], strides=[1] : !flow.dispatch.tensor<readonly:tensor<16xf32>> -> tensor<16xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4, %5 : tensor<16xf32>, tensor<16xf32>) outs(%3 : tensor<16xf32>) {
      ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
          %7 = arith.addf %arg0, %arg1 : f32
          linalg.yield %7 : f32
        } -> tensor<16xf32>
        flow.dispatch.tensor.store %6, %2, offsets=[0], sizes=[16], strides=[1] : tensor<16xf32> -> !flow.dispatch.tensor<writeonly:tensor<16xf32>>
        return
      }
    }
  }
}

// CDNA1-LABEL: hal.executable public @simpleMath_ex_dispatch_0
//       CDNA1:   hal.executable.variant public @rocm
//       CDNA1:   llvm.fadd

// -----

#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0] -> (s0, -d0 + 1024)>
#map2 = affine_map<(d0)[s0] -> (-d0 + 1024, s0)>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @dot_dispatch_0 {
  hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export @dot_dispatch_0 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @dot_dispatch_0() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c1 = arith.constant 1 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<1024x1024xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<1024x1024xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<1024x1024xf32>>
        %8 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1024x1024xf32>> -> tensor<1024x1024xf32>
        %10 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1024x1024xf32>> -> tensor<1024x1024xf32>
        %15 = tensor.empty() : tensor<1024x1024xf32>
        %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
        %17 = linalg.matmul ins(%8, %10 : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
            outs(%16 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
        flow.dispatch.tensor.store %17, %2, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1]
            : tensor<1024x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024x1024xf32>>
        return
      }
    }
  }
}

//   RDNA3-LABEL: hal.executable public @dot_dispatch_0
//         RDNA3:   hal.executable.variant public @rocm
//       RDNA3-NOT:   llvm.store
//           RDNA3:   llvm.br
//   RDNA3-COUNT-1:    llvm.load {{.*}} : !llvm.ptr<1> -> vector<32xf32>
//  RDNA3-COUNT-32:    llvm.load {{.*}} : !llvm.ptr<1> -> vector<16xf32>
//  RDNA3-COUNT-32:    llvm.intr.fmuladd({{.*}}) : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>
//   RDNA3-COUNT-1:    llvm.store {{.*}} : vector<16xf32>, !llvm.ptr<1>
//           RDNA3:   llvm.br

// -----

#map = affine_map<(d0) -> (d0)>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @ext_fp8_dispatch {
  hal.executable.variant @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export @ext_fp8_dispatch layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @ext_fp8_dispatch() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096xf8E4M3FNUZ>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096xf8E5M2FNUZ>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<4096xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [4096], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4096xf8E4M3FNUZ>> -> tensor<4096xf8E4M3FNUZ>
        %4 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [4096], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4096xf8E5M2FNUZ>> -> tensor<4096xf8E5M2FNUZ>
        %5 = tensor.empty() : tensor<4096xf32>
        %6 = linalg.generic {indexing_maps = [#map, #map, #map],
                             iterator_types = ["parallel"]}
                             ins(%3, %4 : tensor<4096xf8E4M3FNUZ>, tensor<4096xf8E5M2FNUZ>)
                             outs(%5 : tensor<4096xf32>) {
        ^bb0(%in0: f8E4M3FNUZ, %in1: f8E5M2FNUZ, %out: f32):
          %7 = arith.extf %in0 : f8E4M3FNUZ to f32
          %8 = arith.extf %in1 : f8E5M2FNUZ to f32
          %9 = arith.addf %7, %8 : f32
          linalg.yield %9 : f32
        } -> tensor<4096xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0], sizes = [4096], strides = [1] : tensor<4096xf32> -> !flow.dispatch.tensor<writeonly:tensor<4096xf32>>
        return
      }
    }
  }
}

//   CDNA3-LABEL: hal.executable public @ext_fp8_dispatch
//         CDNA3:   hal.executable.variant public @rocm
// CDNA3-COUNT-16:     rocdl.cvt.f32.fp8 %{{.*}} : f32
// CDNA3-COUNT-16:     rocdl.cvt.f32.bf8 %{{.*}} : f32
//         CDNA3:     %[[ADD:.+]] = llvm.fadd %{{.*}}, %{{.*}} : vector<16xf32>
//         CDNA3:     llvm.store %[[ADD]], %{{.*}} : vector<16xf32>, !llvm.ptr<1>

// -----

// Verify that the ceildivsi op gets expanded and lowered successfully all the way to
// the llvm dialect.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @ceildiv_expand_dispatch {
  hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export @ceildiv_expand layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @ceildiv_expand() {
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<16xi32>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<16xi32>>
      %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<16xi32>>
      %3 = tensor.empty() : tensor<16xi32>
      %4 = flow.dispatch.tensor.load %0, offsets=[0], sizes=[16], strides=[1] : !flow.dispatch.tensor<readonly:tensor<16xi32>> -> tensor<16xi32>
      %5 = flow.dispatch.tensor.load %1, offsets=[0], sizes=[16], strides=[1] : !flow.dispatch.tensor<readonly:tensor<16xi32>> -> tensor<16xi32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4, %5 : tensor<16xi32>, tensor<16xi32>) outs(%3 : tensor<16xi32>) {
      ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):  // no predecessors
          %7 = arith.ceildivsi %arg0, %arg1 : i32
          linalg.yield %7 : i32
        } -> tensor<16xi32>
        flow.dispatch.tensor.store %6, %2, offsets=[0], sizes=[16], strides=[1] : tensor<16xi32> -> !flow.dispatch.tensor<writeonly:tensor<16xi32>>
        return
      }
    }
  }
}

//   CDNA3-LABEL: hal.executable public @ceildiv_expand_dispatch
//         CDNA3:   hal.executable.variant public @rocm
//     CDNA3-NOT:     arith.ceildivsi
// CDNA3-COUNT-1:     llvm.select {{.*}} : vector<1xi1>, vector<1xi32>
// CDNA3-COUNT-2:     llvm.sdiv {{.*}} : vector<1xi32>
// CDNA3-COUNT-4:     llvm.icmp {{.*}} : vector<1xi32>
// CDNA3-COUNT-2:     llvm.and {{.*}} : vector<1xi1>
// CDNA3-COUNT-1:     llvm.or {{.*}} : vector<1xi1>
// CDNA3-COUNT-1:     llvm.select {{.*}} : vector<1xi1>, vector<1xi32>

// -----
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @erf_dispatch {
  hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
  hal.executable.export @erf layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1
      hal.return %x, %y, %z : index, index, index
    }
builtin.module {
  func.func @erf() {
    %cst = arith.constant 1.270000e+02 : f16
    %cst_0 = arith.constant -1.280000e+02 : f16
    %cst_1 = arith.constant 5.000000e-01 : f16
    %cst_2 = arith.constant 1.000000e+00 : f16
    %cst_3 = arith.constant 2.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)  : !flow.dispatch.tensor<readonly:tensor<2x1024x10240xf16>>
    %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)  : !flow.dispatch.tensor<readonly:tensor<5120xf16>>
    %7 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2)  : !flow.dispatch.tensor<readonly:tensor<f32>>
    %8 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3)  : !flow.dispatch.tensor<writeonly:tensor<2x1024x5120xi8>>
    %9 = flow.dispatch.tensor.load %6, offsets = [0], sizes = [5120], strides = [1] : !flow.dispatch.tensor<readonly:tensor<5120xf16>> -> tensor<5120xf16>
    %10 = flow.dispatch.tensor.load %7, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:tensor<f32>> -> tensor<f32>
    %11 = tensor.empty() : tensor<2x1024x5120xi8>
    %12 = flow.dispatch.tensor.load %5, offsets = [0, 0, 5120], sizes = [2, 1024, 5120], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x1024x10240xf16>> -> tensor<2x1024x5120xf16>
    %13 = flow.dispatch.tensor.load %5, offsets = [0, 0, 0], sizes = [2, 1024, 5120], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x1024x10240xf16>> -> tensor<2x1024x5120xf16>
    %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%13, %12, %9, %10 : tensor<2x1024x5120xf16>, tensor<2x1024x5120xf16>, tensor<5120xf16>, tensor<f32>) outs(%11 : tensor<2x1024x5120xi8>) {
    ^bb0(%in: f16, %in_4: f16, %in_5: f16, %in_6: f32, %out: i8):
      %15 = math.sqrt %cst_3 : f16
      %16 = arith.divf %in_4, %15 : f16
      %17 = math.erf %16 : f16
      %18 = arith.addf %17, %cst_2 : f16
      %19 = arith.mulf %18, %cst_1 : f16
      %20 = arith.mulf %in_4, %19 : f16
      %21 = arith.mulf %in, %20 : f16
      %22 = arith.mulf %21, %in_5 : f16
      %23 = arith.truncf %in_6 : f32 to f16
      %24 = arith.divf %22, %23 : f16
      %25 = math.roundeven %24 : f16
      %26 = arith.cmpf ult, %25, %cst_0 : f16
      %27 = arith.select %26, %cst_0, %25 : f16
      %28 = arith.cmpf ugt, %27, %cst : f16
      %29 = arith.select %28, %cst, %27 : f16
      %30 = arith.fptosi %29 : f16 to i8
      linalg.yield %30 : i8
    } -> tensor<2x1024x5120xi8>
    flow.dispatch.tensor.store %14, %8, offsets = [0, 0, 0], sizes = [2, 1024, 5120], strides = [1, 1, 1] : tensor<2x1024x5120xi8> -> !flow.dispatch.tensor<writeonly:tensor<2x1024x5120xi8>>
    return
  }
}
  }
}

//    CDNA3-LABEL: hal.executable public @erf_dispatch
//          CDNA3:   hal.executable.variant public @rocm
// CDNA3-COUNT-19:     llvm.select {{.*}} : vector<8xi1>, vector<8xf32>
//  CDNA3-COUNT-8:     llvm.intr.fma{{.*}} : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
//          CDNA3:     %[[RESULT:.+]] = llvm.fptosi {{.*}} : vector<8xf16> to vector<8xi8>
//          CDNA3:     llvm.store %[[RESULT]], %{{.*}} : vector<8xi8>, !llvm.ptr<1>
