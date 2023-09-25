// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PJRT_PLUGIN_PJRT_COMMON_DYLIB_PLATFORM_H_
#define IREE_PJRT_PLUGIN_PJRT_COMMON_DYLIB_PLATFORM_H_

#include <optional>
#include <string>

#include "iree_pjrt/common/platform.h"

namespace iree::pjrt {

class DylibPlatform final : public Platform {
 public:
  // Gets the IREE home directory, which is expected to contain
  //   bin/
  //   lib/
  // This is derived from the config key "HOME_DIR";
  std::optional<std::string> GetHomeDir();

  // Gets the directory containing IREE binary tools.
  // This is derived from the config key "BIN_DIR" or computed relative to
  // the home directory.
  std::optional<std::string> GetBinaryDir();

  // Gets the directory containing IREE shared libraries. On Posix platforms,
  // this will typically be under ${IREE_HOME}/lib. On Windows, it is the
  // binary dir.
  // This is derived from the config key "LIB_DIR" or computed relative to
  // the home directory.
  std::optional<std::string> GetLibraryDir();

  // Gets the path to the libIREECompiler.so shared library.
  // This is taken from either the "COMPILER_LIB_PATH" config variable
  // or computed from paths above.
  std::optional<std::string> GetCompilerLibraryPath();

  // Gets the path to the libOpenXLAPartitioner.so shared library.
  // This is taken from either the "PARTITIONER_LIB_PATH" config variable
  // or computed from paths above.
  std::optional<std::string> GetPartitionerLibraryPath();

 protected:
  iree_status_t SubclassInitialize() override;
};

}  // namespace iree::pjrt

#endif  // IREE_PJRT_PLUGIN_PJRT_COMMON_DYLIB_PLATFORM_H_
