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

#ifndef THIRD_PARTY_VULKAN_EXTENSIONLAYER_VULKAN_LOADER_SETUP_H_
#define THIRD_PARTY_VULKAN_EXTENSIONLAYER_VULKAN_LOADER_SETUP_H_

namespace vulkan_extensionlayer {

// Adds the path to the extension layer's manifest file to VK_LAYER_PATH.
//
// For more information about the Vulkan loader and ICD discovery, see the docs:
// https://vulkan.lunarg.com/doc/view/latest/windows/loader_and_layer_interface.html.
void SetupForVulkanLoader();

}  // namespace vulkan_extensionlayer

#endif  // THIRD_PARTY_VULKAN_EXTENSIONLAYER_VULKAN_LOADER_SETUP_H_
