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

#include "iree/hal/vulkan/renderdoc_capture_manager.h"

#include "iree/base/logging.h"
#include "iree/base/platform_headers.h"
#include "iree/base/tracing.h"

#if defined(IREE_PLATFORM_WINDOWS)
#else
#include <dlfcn.h>
#endif  // IREE_PLATFORM_WINDOWS

namespace iree {
namespace hal {
namespace vulkan {

// TODO(scotttodd): absl flag for RTLD_NOLOAD / RTLD_NOW?
//     https://renderdoc.org/docs/in_application_api.html suggests RTLD_NOLOAD

RenderDocCaptureManager::RenderDocCaptureManager() {}

RenderDocCaptureManager::~RenderDocCaptureManager() {
  IREE_TRACE_SCOPE0("RenderDocCaptureManager::dtor");
  Disconnect();
}

Status RenderDocCaptureManager::Connect() {
  IREE_TRACE_SCOPE0("RenderDocCaptureManager::Connect");

  if (renderdoc_library_ != nullptr) {
    return OkStatus();
  }

// Load RenderDoc's library object.
#if defined(IREE_PLATFORM_WINDOWS)
  // TODO(scotttodd): Windows, ::LoadLibraryA()
  return UnimplementedErrorBuilder(IREE_LOC);
#else
  renderdoc_library_ = ::dlopen("librenderdoc.so", RTLD_NOW);
  if (renderdoc_library_ == nullptr) {
    return NotFoundErrorBuilder(IREE_LOC)
           << "Could not load librenderdoc.so. Is it on LD_LIBRARY_PATH?";
  }

  // Fetch the API object from the loaded library.
  pRENDERDOC_GetAPI RENDERDOC_GetAPI =
      (pRENDERDOC_GetAPI)::dlsym(renderdoc_library_, "RENDERDOC_GetAPI");
  int ret =
      RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_4_0, (void**)&renderdoc_api_);
  if (ret != 1) {
    renderdoc_api_ = nullptr;
    return InternalErrorBuilder(IREE_LOC)
           << "Failed to get RenderDoc API object";
  }

  LOG(INFO) << "Connected to RenderDoc's API";

  return OkStatus();
#endif
}

void RenderDocCaptureManager::Disconnect() {
  IREE_TRACE_SCOPE0("RenderDocCaptureManager::Disconnect");

  if (renderdoc_library_ == nullptr) {
    return;
  }

  if (is_capturing()) {
    StopCapture();
  }

  renderdoc_api_ = nullptr;

#if defined(IREE_PLATFORM_WINDOWS)
// TODO(scotttodd): Windows, ::FreeLibrary()
#else
  ::dlclose(renderdoc_library_);
  renderdoc_library_ = nullptr;
#endif
}

void RenderDocCaptureManager::StartCapture() {
  IREE_TRACE_SCOPE0("RenderDocCaptureManager::StartCapture");

  CHECK(is_connected()) << "Can't start capture when not connected";
  CHECK(!is_capturing()) << "Capture is already started";

  LOG(INFO) << "Starting RenderDoc capture";
  renderdoc_api_->StartFrameCapture(NULL, NULL);
}

void RenderDocCaptureManager::StopCapture() {
  IREE_TRACE_SCOPE0("RenderDocCaptureManager::StopCapture");

  CHECK(is_capturing()) << "Can't stop capture when not capturing";

  LOG(INFO) << "Ending RenderDoc capture";
  renderdoc_api_->EndFrameCapture(NULL, NULL);
}

bool RenderDocCaptureManager::is_capturing() const {
  if (!is_connected()) {
    return false;
  }

  return renderdoc_api_->IsFrameCapturing() == 1;
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
