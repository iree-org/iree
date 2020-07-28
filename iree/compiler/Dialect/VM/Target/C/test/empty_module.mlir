// RUN: iree-translate -iree-vm-ir-to-c-module %s | IreeFileCheck %s

// CHECK: #include "vm_c_funcs.h"
vm.module @empty_module {
}
