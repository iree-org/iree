// RUN: iree-translate -iree-vm-ir-to-c-module %s | IreeFileCheck %s

// CHECK: #include "iree/compiler/Dialect/VM/Target/C/vm_c_funcs.h"
vm.module @empty_module {
}
