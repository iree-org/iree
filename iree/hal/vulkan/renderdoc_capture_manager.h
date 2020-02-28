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

#ifndef IREE_HAL_VULKAN_RENDERDOC_CAPTURE_MANAGER_H_
#define IREE_HAL_VULKAN_RENDERDOC_CAPTURE_MANAGER_H_

#include "iree/base/dynamic_library.h"
#include "iree/base/status.h"
#include "iree/hal/debug_capture_manager.h"
#include "third_party/renderdoc_api/app/renderdoc_app.h"

namespace iree {
namespace hal {
namespace vulkan {

// Capture manager using RenderDoc to record Vulkan commands.
// See https://renderdoc.org/ and https://github.com/baldurk/renderdoc.
class RenderDocCaptureManager final : public DebugCaptureManager {
 public:
  RenderDocCaptureManager();
  ~RenderDocCaptureManager() override;

  // Note: Connect() must be called *before* creating a VkInstance.
  Status Connect() override;

  void Disconnect() override;

  bool is_connected() const override { return renderdoc_api_ != nullptr; }

  // Note: StartCapture() must be called *after* creating a VkDevice.
  void StartCapture() override;

  void StopCapture() override;

  bool is_capturing() const override;

 private:
  std::unique_ptr<DynamicLibrary> renderdoc_library_;
  RENDERDOC_API_1_4_0* renderdoc_api_ = nullptr;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_RENDERDOC_CAPTURE_MANAGER_H_
