// RUN: iree-opt --split-input-file --iree-stream-resource-refcounting %s | FileCheck %s

util.global private @__device_0 = #hal.device.target<"hip", {ordinal = 0 : index}, [#hal.executable.target<"rocm", "rocm-hsaco-fb",{}>]> : !hal.device
util.global private @__device_1 = #hal.device.target<"hip", {ordinal = 1 : index}, [#hal.executable.target<"rocm", "rocm-hsaco-fb",{}>]> : !hal.device
util.global private @__device_2 = #hal.device.target<"hip", {ordinal = 2 : index}, [#hal.executable.target<"rocm", "rocm-hsaco-fb",{}>]> : !hal.device
util.global private @__device_3 = #hal.device.target<"hip", {ordinal = 3 : index}, [#hal.executable.target<"rocm", "rocm-hsaco-fb",{}>]> : !hal.device

util.global private @device : !hal.device

// CHECK-LABEL: @locals
// CHECK-SAME: (%[[SIZE0:.+]]: index, %[[SIZE1:.+]]: index, %[[AWAIT_TIMEPOINT:.+]]: !stream.timepoint)
util.func public @locals(%size0: index, %size1: index, %await_timepoint: !stream.timepoint,%arg0: !stream.resource<constant>, %arg2: !stream.resource<constant>, %arg1: index,  %arg3: index) -> !stream.timepoint, !stream.timepoint{
  %c254_i32 = arith.constant 254 : i32
  %c255_i32 = arith.constant 255 : i32
  %source_size = arith.constant 256 : index
  %target_size = arith.constant 256 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index

 %c67108864 = arith.constant 67108864 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1048576 = arith.constant 1048576 : index

%results:4, %result_timepoint = stream.async.execute on(#hal.device.affinity<@__device_0>) with(%arg0 as %arg13: !stream.resource<constant>{%c1048576}, %arg2 as %arg14: !stream.resource<constant>{%c512}) -> (!stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864},!stream.resource<transient>{%c67108864}) {

    %581 = stream.async.dispatch @xyz(%arg13[%c0 to %c1048576 for %c1048576], %arg14[%c0 to %c512 for %c512]) : (!stream.resource<constant>{%c1048576}, !stream.resource<constant>{%c512}) -> !stream.resource<transient>{%c67108864}

    %582:3 = stream.async.concurrent with(%581 as %arg15: !stream.resource<transient>{%c67108864}) -> (!stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864}) {
      %200 = stream.async.transfer %arg15 : !stream.resource<transient>{%c67108864} -> to(#hal.device.affinity<@__device_1>) !stream.resource<transient>{%c67108864}
      %201 = stream.async.transfer %arg15 : !stream.resource<transient>{%c67108864} -> to(#hal.device.affinity<@__device_2>) !stream.resource<transient>{%c67108864}
      %202 = stream.async.transfer %arg15 : !stream.resource<transient>{%c67108864} -> to(#hal.device.affinity<@__device_3>) !stream.resource<transient>{%c67108864}

      stream.yield %200, %201, %202 : !stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864}
    }

    stream.yield %581, %582#0, %582#1,%582#2  : !stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864}
    } => !stream.timepoint

    %results_0:8, %result_timepoint_1 = stream.async.execute on(#hal.device.affinity<@__device_1>)  await(%result_timepoint) =>  with(%arg0 as %arg13: !stream.resource<constant>{%c1048576}, %arg2 as %arg14: !stream.resource<constant>{%c512}, %results#1 as %arg20 : !stream.resource<transient>{%c67108864}) -> (!stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864},!stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864},!stream.resource<transient>{%c67108864},!stream.resource<transient>{%c67108864},!stream.resource<transient>{%c67108864}) {
      %581 = stream.async.transfer %arg13 : !stream.resource<constant>{%c1048576} from(#hal.device.affinity<@__device_0>) -> !stream.resource<transient>{%c1048576}

      %582:2 = stream.async.concurrent with(%arg14 as %arg23: !stream.resource<constant>{%c512}, %arg13 as %arg24: !stream.resource<constant>{%c1048576}, %581 as %arg25: !stream.resource<transient>{%c1048576}) -> (!stream.resource<transient>{%c1048576}, !stream.resource<transient>{%c67108864}) {
            %590 = stream.async.transfer %arg23 : !stream.resource<constant>{%c512} from(#hal.device.affinity<@__device_0>) -> !stream.resource<transient>{%c1048576}
            %591 = stream.async.dispatch @abcd(%arg24[%c0 to %c1048576 for %c1048576], %arg25[%c0 to %c512 for %c512]) : (!stream.resource<constant>{%c1048576}, !stream.resource<transient>{%c1048576}) -> !stream.resource<transient>{%c67108864}
      stream.yield %590, %591 : !stream.resource<transient>{%c1048576}, !stream.resource<transient>{%c67108864}
      }

    %583 = stream.async.dispatch @abc(%582#1[%c0 to %c67108864 for %c67108864]): (!stream.resource<transient>{%c67108864}) -> !stream.resource<transient>{%c67108864}
    %584:2 = stream.async.concurrent with(%583 as %arg23: !stream.resource<transient>{%c67108864}) -> (!stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864}){
        %590 = stream.async.dispatch @dispatch_16(%arg23[%c0 to %c67108864 for %c67108864]) : (!stream.resource<transient>{%c67108864}) -> !stream.resource<transient>{%c67108864}
        %591 = stream.async.dispatch @dispatch_24(%arg23[%c0 to %c67108864 for %c67108864]) : (!stream.resource<transient>{%c67108864}) -> !stream.resource<transient>{%c67108864}
    stream.yield %590, %591 : !stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864}
    }

    %585:4 = stream.async.concurrent with(%583 as %arg23: !stream.resource<transient>{%c67108864}, %arg20 as %arg26: !stream.resource<transient>{%c67108864}, %584#1 as %arg25: !stream.resource<transient>{%c67108864}, %582#0 as %arg27: !stream.resource<transient>{%c1048576}) -> (!stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864}) {
     %590 =stream.async.dispatch @dispatch_24(%arg23[%c0 to %c67108864 for %c67108864]) : (!stream.resource<transient>{%c67108864}) -> !stream.resource<transient>{%c67108864}

    %591:3 = stream.async.dispatch @abc_scatter(%arg25[%c0 to %c67108864 for %c67108864], %arg26[%c0 to %c512 for %c512], %arg27[%c0 to %c512 for %c512]) : (!stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c512},!stream.resource<transient>{%c512}) -> (!stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864})
      stream.yield %590, %591#0, %591#1,%591#2 : !stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864}
    }

    %586:2 = stream.async.concurrent with(%585#3 as %arg23: !stream.resource<transient>{%c67108864}, %585#0 as %arg24: !stream.resource<transient>{%c67108864}) -> (!stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864}) {
    %590 = stream.async.dispatch @dispatch_57(%arg23[%c0 to %c512 for %c512]) : (!stream.resource<transient>{%c512}) -> (!stream.resource<transient>{%c512})
     %591 = stream.async.dispatch @dispatch_broadcast(%arg24[%c0 to %c67108864 for %c67108864]) : (!stream.resource<transient>{%c67108864}) -> !stream.resource<transient>{%c67108864}
     stream.yield %590, %591 : !stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864}
    }

    %587 = stream.async.dispatch @dispatch_73(%arg20[%c0 to %c67108864 for %c67108864], %584#0[%c0 to %c512 for %c512], %586#0[%c0 to %c512 for %c512], %586#1[%c0 to %c512 for %c512]) : (!stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864}) -> !stream.resource<transient>{%c67108864}
    %588 = stream.async.dispatch @dispatch_81(%587[%c0 to %c512 for %c512]) : (!stream.resource<transient>{%c67108864}) -> !stream.resource<transient>{%c67108864}
    %589:3 = stream.async.concurrent with(%588 as %arg23: !stream.resource<transient>{%c67108864}) -> (!stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864}){
        %590 = stream.async.transfer %arg23 : !stream.resource<transient>{%c67108864} -> to(#hal.device.affinity<@__device_0>) !stream.resource<transient>{%c67108864}
        %591 = stream.async.transfer %arg23 : !stream.resource<transient>{%c67108864} -> to(#hal.device.affinity<@__device_2>) !stream.resource<transient>{%c67108864}
        %592 = stream.async.transfer %arg23 : !stream.resource<transient>{%c67108864} -> to(#hal.device.affinity<@__device_3>) !stream.resource<transient>{%c67108864}
        stream.yield %590, %591, %592 : !stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864}
    }

    stream.yield %582#1, %585#0, %585#1, %585#2, %588, %589#0, %589#1, %589#2 : !stream.resource<transient>{%c67108864}, !stream.resource<transient>{%c67108864},!stream.resource<transient>{%c67108864},!stream.resource<transient>{%c67108864},!stream.resource<transient>{%c67108864},!stream.resource<transient>{%c67108864},!stream.resource<transient>{%c67108864},!stream.resource<transient>{%c67108864}
} => !stream.timepoint

  util.return %result_timepoint, %result_timepoint_1: !stream.timepoint, !stream.timepoint
}
