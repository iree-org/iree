// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PJRT_PLUGIN_PJRT_PLATFORM_H_
#define IREE_PJRT_PLUGIN_PJRT_PLATFORM_H_

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

#include "iree/base/status.h"
#include "iree/integrations/pjrt/common/compiler.h"
#include "iree/integrations/pjrt/common/debugging.h"

namespace iree::pjrt {

//===----------------------------------------------------------------------===//
// ConfigVars
// Placeholder for API-level configuration (i.e. from environment, files, etc).
//===----------------------------------------------------------------------===//

struct ConfigVars {
 public:
  ConfigVars() = default;
  // Configures this instance to fallback to the system environment if a config
  // var is not found by prefixing |env_fallback_prefix| to the variable name.
  void EnableEnvFallback(std::string env_fallback_prefix);

  // Looks up a variable value by name. On success, the resulting string_view
  // is valid at least until the next mutation.
  std::optional<std::string> Lookup(const std::string& key);

 private:
  std::unordered_map<std::string, std::string> kv_entries_;
  std::optional<std::string> env_fallback_prefix_;
};

//===----------------------------------------------------------------------===//
// Platform
// Encapsulates aspects of the platform which may differ under different
// usage and/or environments.
//===----------------------------------------------------------------------===//

class Platform {
 public:
  virtual ~Platform();
  iree_status_t Initialize();
  ConfigVars& config_vars() { return config_vars_; }
  Logger& logger() { return *logger_; }
  AbstractCompiler& compiler() { return *compiler_; }
  ArtifactDumper& artifact_dumper() { return *artifact_dumper_; }

 protected:
  virtual iree_status_t SubclassInitialize() = 0;
  ConfigVars config_vars_;
  std::unique_ptr<Logger> logger_;
  std::unique_ptr<AbstractCompiler> compiler_;
  std::unique_ptr<ArtifactDumper> artifact_dumper_;
};

}  // namespace iree::pjrt

#endif  // IREE_PJRT_PLUGIN_PJRT_PLATFORM_H_
