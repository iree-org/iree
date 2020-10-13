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

#ifndef IREE_HAL_METAL_METAL_CAPTURE_MANAGER_H_
#define IREE_HAL_METAL_METAL_CAPTURE_MANAGER_H_

#include <memory>

#import <Metal/Metal.h>

#include "iree/base/status.h"
#include "iree/hal/debug_capture_manager.h"

namespace iree {
namespace hal {
namespace metal {

// A DebugCaptureManager implementation for Metal that directly wraps a
// MTLCaptureManager.
class MetalCaptureManager final : public DebugCaptureManager {
 public:
  // Creates a capture manager that captures Metal commands to the given |capture_file| if not
  // empty. Capture to Xcode otherwise.
  static StatusOr<std::unique_ptr<MetalCaptureManager>> Create(const std::string& capture_file);
  ~MetalCaptureManager() override;

  Status Connect() override;

  void Disconnect() override;

  bool is_connected() const override;

  void SetCaptureObject(id object);

  void StartCapture() override;

  void StopCapture() override;

  bool is_capturing() const override;

 private:
  explicit MetalCaptureManager(NSURL* capture_file);

  MTLCaptureManager* metal_handle_ = nil;
  // The path for storing the .gputrace file. Empty means capturing to Xcode.
  NSURL* capture_file_ = nil;
  id capture_object_ = nil;
};

}  // namespace metal
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_METAL_METAL_CAPTURE_MANAGER_H_
