// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=hip},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-hip-target=gfx942 --iree-hip-enable-experimental-pad-layout %s | FileCheck %s --check-prefix=PAD
//
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=hip},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-hip-target=gfx90a --iree-hip-enable-experimental-pad-layout %s | FileCheck %s --check-prefix=PAD

// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=hip},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-hip-target=gfx90a --iree-hip-enable-experimental-pad-layout=false %s | FileCheck %s --check-prefix=NOPAD

// PAD:      #hal.executable.target<"rocm"
// PAD-SAME:   iree.encoding.resolver = #iree_gpu.gpu_pad_layout<cache_line_bytes = 128, cache_sets = 4>

// NOPAD:      #hal.executable.target<"rocm"
// NOPAD-NOT:    iree.encoding.resolver = #iree_gpu.gpu_pad_layout

stream.executable public @main {
  stream.executable.export @main workgroups(%arg0: index) -> (index, index, index) {
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg0
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @main() {
      return
    }
  }
}
