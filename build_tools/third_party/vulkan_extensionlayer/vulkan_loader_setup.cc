// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "vulkan_loader_setup.h"

#include <stdlib.h>

#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "iree/base/initializer.h"
#include "iree/base/logging.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace vulkan_extensionlayer {

namespace {
using bazel::tools::cpp::runfiles::Runfiles;
}  // namespace

void SetupForVulkanLoader() {
  // Get the 'runfiles' path (in Bazel) for the current program.
  std::string error;
  // Note: 'program_invocation_name' is a global variable set to the value of
  //       argv[0] in main(). This only works on some systems.
  std::unique_ptr<Runfiles> runfiles(
      Runfiles::Create(/*argv0=*/program_invocation_name, &error));
  if (!runfiles) {
    LOG(ERROR)
        << "Failed to find runfiles directory for vulkan_extensionlayer: "
        << error;
    return;
  }

  // Find 'vulkan_extensionlayer/' under the runfiles directory.
  std::string path_to_manifest = runfiles->Rlocation("vulkan_extensionlayer");
  if (path_to_manifest.empty()) {
    LOG(ERROR) << "Failed to find runfiles path for vulkan_extensionlayer";
    return;
  }

  // Append the path to the folder with the manifest file to VK_LAYER_PATH.
  char* original_vk_layer_path;
  original_vk_layer_path = getenv("VK_LAYER_PATH");
  std::string vk_layer_path;
  if (original_vk_layer_path) {
    vk_layer_path = absl::StrCat(original_vk_layer_path, ":", path_to_manifest);
  } else {
    vk_layer_path = path_to_manifest;
  }
  setenv("VK_LAYER_PATH", vk_layer_path.c_str(), 0);
}

}  // namespace vulkan_extensionlayer

IREE_DECLARE_MODULE_INITIALIZER(iree_vulkan_extensionlayer);
IREE_REGISTER_MODULE_INITIALIZER(iree_vulkan_extensionlayer,
                                 vulkan_extensionlayer::SetupForVulkanLoader());
