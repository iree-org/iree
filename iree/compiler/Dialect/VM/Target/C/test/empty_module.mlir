// RUN: iree-translate -iree-vm-ir-to-c-module %s | IreeFileCheck %s

// CHECK: // c module stub
vm.module @empty_module {
}
