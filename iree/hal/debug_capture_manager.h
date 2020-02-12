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

#ifndef IREE_HAL_DEBUG_CAPTURE_MANAGER_H_
#define IREE_HAL_DEBUG_CAPTURE_MANAGER_H_

#include "iree/base/status.h"

namespace iree {
namespace hal {

// Interface for interacting with command recorders / debuggers.
//
// Subclasses connect to tools like RenderDoc or MTLCaptureManager and use them
// to record commands sent to underlying APIs like Vulkan or Metal, for future
// debugging and analysis.
class DebugCaptureManager {
 public:
  DebugCaptureManager() {}
  virtual ~DebugCaptureManager() = default;

  // Attempts to connect to a command recorder, if not already connected.
  //
  // This should be called *before* the underlying system and its devices (such
  // as a VkInstance and its VkDevices) are initialized, so the command recorder
  // can inject any necessary hooks.
  virtual Status Connect() = 0;

  // Disconnects from a connected command recorder, if connected.
  // This implicitly stops capture if currently capturing.
  virtual void Disconnect() = 0;

  // Returns true if connected to a command recorder.
  virtual bool is_connected() const = 0;

  // Starts capturing commands.
  // Must already be connected and must not already be capturing.
  virtual void StartCapture() = 0;

  // Stops capturing commands and saves the capture.
  // Must already be connected and capturing.
  virtual void StopCapture() = 0;

  // Returns true if currently capturing commands.
  virtual bool is_capturing() const = 0;
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_DEBUG_CAPTURE_MANAGER_H_
