// clang-format off
// RUN: mkdir -p %t
// RUN: clang -fPIC -shared -o %t/libinvalid_plugin.so %s
// RUN: (iree-opt --iree-load-plugin=invalid_plugin=%t/libinvalid_plugin.so 2>&1 || true) | FileCheck %s --check-prefix=PLUGIN_LOAD_FAIL
// clang-format on

#include <stdbool.h>

bool some_function() { return false; }

// clang-format off
// PLUGIN_LOAD_FAIL: [IREE Dynamic Plugin ERROR]: could not find registration function 'iree_register_compiler_plugin_invalid_plugin' in plugin
// clang-format on
