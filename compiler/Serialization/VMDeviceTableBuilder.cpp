// Copyright 2019 Google LLC
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

#include "compiler/Serialization/VMDeviceTableBuilder.h"

namespace mlir {
namespace iree_compiler {

VMDeviceTableBuilder::VMDeviceTableBuilder(
    ::flatbuffers::FlatBufferBuilder *fbb)
    : fbb_(fbb) {}

LogicalResult VMDeviceTableBuilder::AddDevice(
    ::flatbuffers::Offset<iree::DeviceDef> deviceDef) {
  deviceDefs_.push_back(deviceDef);
  return success();
}

LogicalResult VMDeviceTableBuilder::AddDeviceGroup(
    ::flatbuffers::Offset<iree::DeviceGroupDef> deviceGroupDef) {
  deviceGroupDefs_.push_back(deviceGroupDef);
  return success();
}

::flatbuffers::Offset<iree::DeviceTableDef> VMDeviceTableBuilder::Finish() {
  auto devicesOffset = fbb_->CreateVector(deviceDefs_);
  auto deviceGroupsOffset = fbb_->CreateVector(deviceGroupDefs_);
  iree::DeviceTableDefBuilder dtdb(*fbb_);
  dtdb.add_devices(devicesOffset);
  dtdb.add_device_groups(deviceGroupsOffset);
  return dtdb.Finish();
}

}  // namespace iree_compiler
}  // namespace mlir
