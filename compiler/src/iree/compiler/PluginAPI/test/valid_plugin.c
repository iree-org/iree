// clang-format off
// RUN: mkdir -p %t
// RUN: %host_cc -fPIC -shared -o %t/libvalid_plugin.so %s
// RUN: iree-opt --iree-load-plugin=%t/libvalid_plugin.so --help
// clang-format on

#include <stdbool.h>

struct IreeCompilerPluginInfo {
  int apiVersion;
  const char *pluginId;
  bool (*registerPlugin)(void *registrar);
};

static bool register_sample_plugin(void *registrar) {
  (void)registrar;
  return true;
}

struct IreeCompilerPluginInfo iree_get_compiler_plugin_info(void) {
  struct IreeCompilerPluginInfo info = {1, "sample_plugin",
                                        register_sample_plugin};
  return info;
}
