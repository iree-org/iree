// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-stream-dump-statistics{output-format=pretty})" %s 2>&1 | FileCheck %s --check-prefix=CHECK-PRETTY
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-stream-dump-statistics{output-format=csv})" %s 2>&1 | FileCheck %s --check-prefix=CHECK-CSV

// CHECK-PRETTY: Aggregate Statistics
// CHECK-PRETTY:   Constants: 1, 0 B
// CHECK-PRETTY:   Variables: 0, 0 B
// CHECK-PRETTY:  D->H Syncs: 2
// CHECK-PRETTY: Submissions: 3, using cumulative 0 B
// CHECK-PRETTY:   DMA Fills: 0
// CHECK-PRETTY:  DMA Copies: 2
// CHECK-PRETTY: Collectives: 0
// CHECK-PRETTY:  Dispatches: 3
// CHECK-PRETTY: Executables: 2, 33% reuse

// CHECK-CSV: ; Aggregate Statistics
// CHECK-CSV: "Constants","Constant Size","Variables","Variable Size","Awaits","Submissions","Transient Size","Fills","Copies","Dispatches","Async Calls","Executables"
// CHECK-CSV: 1,0,0,0,2,3,0,0,2,3,0,2
// CHECK-CSV: ; Execution
// CHECK-CSV: "Depth","Command","Symbol","Length","Invocations","Workload","Operands","Resources"
// CHECK-CSV: 0,"copy",,192,,,,
// CHECK-CSV: 0,"dispatch","@func_a_ex_0::@dispatch_0",,4,"4;1;1",0,3

util.global private mutable @_constant__timepoint = #stream.timepoint<immediate>
util.global private @_constant : !stream.resource<constant>
util.initializer {
  %c0 = arith.constant 0 : index
  %c192 = arith.constant 192 : index
  %0 = stream.timepoint.immediate => !stream.timepoint
  %1 = util.buffer.constant {alignment = 32 : index} : !util.buffer = #util.composite<192xi8, [
      dense<[5, 6, 7, 8]> : tensor<4xi32>,
      dense<0> : vector<16xi8>,
      dense<[5, 6, 3, 8]> : tensor<4xi32>,
      dense<0> : vector<16xi8>,
      dense<[1, 6, 7, 8]> : tensor<4xi32>,
      dense<0> : vector<16xi8>,
      dense<[5, 6, 7]> : tensor<3xi32>,
      dense<0> : vector<20xi8>,
      dense<[5, 6, 3]> : tensor<3xi32>,
      dense<0> : vector<20xi8>,
      dense<[1, 6, 7]> : tensor<3xi32>,
      dense<0> : vector<20xi8>,
  ]>
  %did_map, %result = stream.resource.try_map %1[%c0] : !util.buffer -> i1, !stream.resource<constant>{%c192}
  %2:2 = scf.if %did_map -> (!stream.resource<constant>, !stream.timepoint) {
    scf.yield %result, %0 : !stream.resource<constant>, !stream.timepoint
  } else {
    %3 = stream.resource.map %1[%c0] : !util.buffer -> !stream.resource<staging>{%c192}
    %4 = stream.resource.alloc uninitialized : !stream.resource<constant>{%c192}
    %5 = stream.cmd.execute with(%3 as %arg0: !stream.resource<staging>{%c192}, %4 as %arg1: !stream.resource<constant>{%c192}) {
      stream.cmd.copy %arg0[%c0], %arg1[%c0], %c192 : !stream.resource<staging>{%c192} -> !stream.resource<constant>{%c192}
    } => !stream.timepoint
    scf.yield %4, %5 : !stream.resource<constant>, !stream.timepoint
  }
  util.global.store %2#0, @_constant : !stream.resource<constant>
  util.global.store %2#1, @_constant__timepoint : !stream.timepoint
  util.initializer.return
}

stream.executable private @func_a_ex_0 {
  stream.executable.export public @dispatch_0
  builtin.module {
    func.func @dispatch_0(%arg0: !stream.binding {stream.alignment = 32 : index}, %arg1: !stream.binding {stream.alignment = 32 : index}, %arg2: !stream.binding {stream.alignment = 32 : index}) {
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<4xi32>>
      %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<4xi32>>
      %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<4xi32>>
      %workgroup_size_0 = flow.dispatch.workgroup.size[0] : index
      %workgroup_id_0 = flow.dispatch.workgroup.id[0] : index
      %workgroup_count_0 = flow.dispatch.workgroup.count[0] : index
      %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_0, %workgroup_size_0]
      %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_0, %workgroup_size_0]
      scf.for %arg3 = %3 to %c4 step %4 {
        %5 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 4)>(%arg3)[%workgroup_size_0]
        %6 = flow.dispatch.tensor.load %0, offsets = [%arg3], sizes = [%5], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4xi32>> -> tensor<?xi32>
        %7 = flow.dispatch.tensor.load %1, offsets = [%arg3], sizes = [%5], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4xi32>> -> tensor<?xi32>
        %8 = tensor.empty(%5) : tensor<?xi32>
        %9 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%6, %7 : tensor<?xi32>, tensor<?xi32>) outs(%8 : tensor<?xi32>) {
        ^bb0(%arg4: i32, %arg5: i32, %arg6: i32):  // no predecessors
          %10 = arith.maxsi %arg4, %arg5 : i32
          linalg.yield %10 : i32
        } -> tensor<?xi32>
        flow.dispatch.tensor.store %9, %2, offsets = [%arg3], sizes = [%5], strides = [1] : tensor<?xi32> -> !flow.dispatch.tensor<writeonly:tensor<4xi32>>
      }
      return
    }
  }
}

stream.executable private @func_a_ex_1 {
  stream.executable.export public @dispatch_1
  builtin.module {
    func.func @dispatch_1(%arg0: !stream.binding {stream.alignment = 32 : index}, %arg1: !stream.binding {stream.alignment = 32 : index}, %arg2: !stream.binding {stream.alignment = 32 : index}) {
      %c3 = arith.constant 3 : index
      %c0 = arith.constant 0 : index
      %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<3xi32>>
      %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<3xi32>>
      %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<3xi32>>
      %workgroup_size_0 = flow.dispatch.workgroup.size[0] : index
      %workgroup_id_0 = flow.dispatch.workgroup.id[0] : index
      %workgroup_count_0 = flow.dispatch.workgroup.count[0] : index
      %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_0, %workgroup_size_0]
      %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_0, %workgroup_size_0]
      scf.for %arg3 = %3 to %c3 step %4 {
        %5 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 3)>(%arg3)[%workgroup_size_0]
        %6 = flow.dispatch.tensor.load %0, offsets = [%arg3], sizes = [%5], strides = [1] : !flow.dispatch.tensor<readonly:tensor<3xi32>> -> tensor<?xi32>
        %7 = flow.dispatch.tensor.load %1, offsets = [%arg3], sizes = [%5], strides = [1] : !flow.dispatch.tensor<readonly:tensor<3xi32>> -> tensor<?xi32>
        %8 = tensor.empty(%5) : tensor<?xi32>
        %9 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%6, %7 : tensor<?xi32>, tensor<?xi32>) outs(%8 : tensor<?xi32>) {
        ^bb0(%arg4: i32, %arg5: i32, %arg6: i32):  // no predecessors
          %10 = arith.maxsi %arg4, %arg5 : i32
          linalg.yield %10 : i32
        } -> tensor<?xi32>
        flow.dispatch.tensor.store %9, %2, offsets = [%arg3], sizes = [%5], strides = [1] : tensor<?xi32> -> !flow.dispatch.tensor<writeonly:tensor<3xi32>>
      }
      return
    }
  }
}

func.func public @func_a() -> (tensor<4xi32>, tensor<4xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c192 = arith.constant 192 : index
  %_constant__timepoint = util.global.load @_constant__timepoint : !stream.timepoint
  %_constant = util.global.load @_constant : !stream.resource<constant>
  %0 = stream.resource.alloc uninitialized : !stream.resource<external>{%c16}
  %1 = stream.cmd.execute await(%_constant__timepoint) => with(%_constant as %arg0: !stream.resource<constant>{%c192}, %0 as %arg1: !stream.resource<external>{%c16}) {
    stream.cmd.copy %arg0[%c0], %arg1[%c0], %c16 : !stream.resource<constant>{%c192} -> !stream.resource<external>{%c16}
  } => !stream.timepoint
  %2 = stream.resource.alloc uninitialized : !stream.resource<external>{%c16}
  %3 = stream.cmd.execute await(%_constant__timepoint) => with(%_constant as %arg0: !stream.resource<constant>{%c192}, %2 as %arg1: !stream.resource<external>{%c16}) {
    stream.cmd.dispatch @func_a_ex_0::@dispatch_0[%c4, %c1, %c1] {
      ro %arg0[%c64 for %c16] : !stream.resource<constant>{%c192},
      ro %arg0[%c32 for %c16] : !stream.resource<constant>{%c192},
      wo %arg1[%c0 for %c16] : !stream.resource<external>{%c16}
    }
    stream.cmd.dispatch @func_a_ex_0::@dispatch_0[%c4, %c1, %c1] {
      ro %arg0[%c64 for %c16] : !stream.resource<constant>{%c192},
      ro %arg0[%c32 for %c16] : !stream.resource<constant>{%c192},
      wo %arg1[%c0 for %c16] : !stream.resource<external>{%c16}
    }
    stream.cmd.dispatch @func_a_ex_1::@dispatch_1[%c4, %c1, %c1] {
      ro %arg0[%c64 for %c16] : !stream.resource<constant>{%c192},
      ro %arg0[%c32 for %c16] : !stream.resource<constant>{%c192},
      wo %arg1[%c0 for %c16] : !stream.resource<external>{%c16}
    }
  } => !stream.timepoint
  %4 = stream.timepoint.await %3 => %2 : !stream.resource<external>{%c16}
  %5 = stream.tensor.export %4 : tensor<4xi32> in !stream.resource<external>{%c16} -> tensor<4xi32>
  %6 = stream.timepoint.await %1 => %0 : !stream.resource<external>{%c16}
  %7 = stream.tensor.export %6 : tensor<4xi32> in !stream.resource<external>{%c16} -> tensor<4xi32>
  return %5, %7 : tensor<4xi32>, tensor<4xi32>
}
