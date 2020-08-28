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

#include "iree/hal/metal/metal_capture_manager.h"

#include <string>

#include "absl/flags/flag.h"
#include "iree/base/file_io.h"
#include "iree/base/logging.h"
#include "iree/base/tracing.h"

ABSL_FLAG(std::string, metal_capture_to_file, "",
          "Full path to store the GPU trace file (empty means capture to Xcode)");

namespace iree {
namespace hal {
namespace metal {

// static
StatusOr<std::unique_ptr<MetalCaptureManager>> MetalCaptureManager::Create() {
  IREE_TRACE_SCOPE0("MetalCaptureManager::Create");
  @autoreleasepool {
    NSURL* capture_url = nil;
    std::string cpp_path = absl::GetFlag(FLAGS_metal_capture_to_file);
    if (!cpp_path.empty()) {
      NSString* ns_string = [NSString stringWithCString:cpp_path.c_str()
                                               encoding:[NSString defaultCStringEncoding]];
      NSString* capture_path = ns_string.stringByStandardizingPath;
      capture_url = [[NSURL fileURLWithPath:capture_path isDirectory:false] retain];
    }
    return absl::WrapUnique(new MetalCaptureManager(capture_url));
  }
}

MetalCaptureManager::MetalCaptureManager(NSURL* capture_file) : capture_file_(capture_file) {}

MetalCaptureManager::~MetalCaptureManager() {
  IREE_TRACE_SCOPE0("MetalCaptureManager::dtor");
  Disconnect();
  if (capture_file_) [capture_file_ release];
}

Status MetalCaptureManager::Connect() {
  IREE_TRACE_SCOPE0("MetalCaptureManager::Connect");

  if (metal_handle_) return OkStatus();

  @autoreleasepool {
    metal_handle_ = [[MTLCaptureManager sharedCaptureManager] retain];

    if (capture_file_ &&
        [metal_handle_ supportsDestination:MTLCaptureDestinationGPUTraceDocument]) {
      IREE_LOG(INFO) << "Connected to shared Metal capture manager; writing capture to "
                     << std::string([capture_file_.absoluteString UTF8String]);
    } else {
      IREE_LOG(INFO) << "Connected to shared Metal capture manager; capturing to Xcode";
    }
  }

  return OkStatus();
}

void MetalCaptureManager::Disconnect() {
  IREE_TRACE_SCOPE0("MetalCaptureManager::Disconnect");

  if (!metal_handle_) return;

  if (is_capturing()) StopCapture();

  [metal_handle_ release];
  metal_handle_ = nil;
}

bool MetalCaptureManager::is_connected() const { return metal_handle_ != nil; }

void MetalCaptureManager::SetCaptureObject(id object) { capture_object_ = object; }

void MetalCaptureManager::StartCapture() {
  IREE_TRACE_SCOPE0("MetalCaptureManager::StartCapture");

  IREE_CHECK(is_connected()) << "Can't start capture when not connected";
  IREE_CHECK(!is_capturing()) << "Capture is already started";
  IREE_CHECK(capture_object_) << "Must set capture object before starting";

  IREE_LOG(INFO) << "Starting Metal capture";
  @autoreleasepool {
    MTLCaptureDescriptor* capture_descriptor = [[[MTLCaptureDescriptor alloc] init] autorelease];
    capture_descriptor.captureObject = capture_object_;
    if (capture_file_) {
      capture_descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
      capture_descriptor.outputURL = capture_file_;
    } else {
      capture_descriptor.destination = MTLCaptureDestinationDeveloperTools;
    }

    NSError* error;
    if (![metal_handle_ startCaptureWithDescriptor:capture_descriptor error:&error]) {
      NSLog(@"Failed to start capture, error %@", error);
    }
  }
}

void MetalCaptureManager::StopCapture() {
  IREE_TRACE_SCOPE0("MetalCaptureManager::StopCapture");

  IREE_CHECK(is_capturing()) << "Can't stop capture when not capturing";

  IREE_LOG(INFO) << "Ending Metal capture";
  [metal_handle_ stopCapture];
}

bool MetalCaptureManager::is_capturing() const {
  if (!is_connected()) return false;
  return metal_handle_.isCapturing;
}

}  // namespace metal
}  // namespace hal
}  // namespace iree
