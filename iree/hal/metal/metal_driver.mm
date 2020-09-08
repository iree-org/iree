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

#include "iree/hal/metal/metal_driver.h"

#import <Metal/Metal.h>

#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/metal/metal_device.h"

namespace iree {
namespace hal {
namespace metal {

namespace {

// Returns an autoreleased array of available Metal GPU devices.
NSArray<id<MTLDevice>>* GetAvailableMetalDevices() {
#if defined(IREE_PLATFORM_MACOS)
  // For macOS, we might have more than one GPU devices.
  return [MTLCopyAllDevices() autorelease];
#else
  // For other Apple platforms, we only have one GPU device.
  id<MTLDevice> device = [MTLCreateSystemDefaultDevice() autorelease];
  return [NSArray arrayWithObject:device];
#endif
}

}  // namespace

// static
StatusOr<ref_ptr<MetalDriver>> MetalDriver::Create() {
  IREE_TRACE_SCOPE0("MetalDriver::Create");

  @autoreleasepool {
    NSArray<id<MTLDevice>>* devices = GetAvailableMetalDevices();
    if (devices == nil) {
      return UnavailableErrorBuilder(IREE_LOC) << "no Metal GPU devices available";
    }

    std::vector<DeviceInfo> device_infos;
    for (id<MTLDevice> device in devices) {
      std::string name = std::string([device.name UTF8String]);
      DeviceFeatureBitfield supported_features = DeviceFeature::kNone;
      DriverDeviceID device_id = reinterpret_cast<DriverDeviceID>((__bridge void*)device);
      device_infos.emplace_back("metal", std::move(name), supported_features, device_id);
    }
    return assign_ref(new MetalDriver(std::move(device_infos)));
  }
}

MetalDriver::MetalDriver(std::vector<DeviceInfo> devices)
    : Driver("metal"), devices_(std::move(devices)) {
  // Retain all the retained Metal GPU devices.
  for (const auto& device : devices_) {
    [(__bridge id<MTLDevice>)device.device_id() retain];
  }
}

MetalDriver::~MetalDriver() {
  IREE_TRACE_SCOPE0("MetalDriver::dtor");

  // Release all the retained Metal GPU devices.
  for (const auto& device : devices_) {
    [(__bridge id<MTLDevice>)device.device_id() release];
  }
}

StatusOr<std::vector<DeviceInfo>> MetalDriver::EnumerateAvailableDevices() {
  IREE_TRACE_SCOPE0("MetalDriver::EnumerateAvailableDevices");

  return devices_;
}

StatusOr<ref_ptr<Device>> MetalDriver::CreateDefaultDevice() {
  IREE_TRACE_SCOPE0("MetalDriver::CreateDefaultDevice");

  if (devices_.empty()) {
    return UnavailableErrorBuilder(IREE_LOC) << "no Metal GPU devices available";
  }
  return CreateDevice(devices_.front().device_id());
}

StatusOr<ref_ptr<Device>> MetalDriver::CreateDevice(DriverDeviceID device_id) {
  IREE_TRACE_SCOPE0("MetalDriver::CreateDevice");

  for (const DeviceInfo& info : devices_) {
    if (info.device_id() == device_id) return MetalDevice::Create(add_ref(this), info);
  }
  return InvalidArgumentErrorBuilder(IREE_LOC) << "unknown driver device id: " << device_id;
}

}  // namespace metal
}  // namespace hal
}  // namespace iree
