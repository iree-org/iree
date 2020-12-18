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

#ifndef IREE_REMOTING_PROTOCOL_V1_HAL_STUB_H_
#define IREE_REMOTING_PROTOCOL_V1_HAL_STUB_H_

#include "experimental/remoting/iree/remoting/protocol_v1/handler.h"

namespace iree {
namespace remoting {
namespace protocol_v1 {

// Stub for marshalling HAL client invocations over the remoting layer.
class HalClientStub {
 public:
  HalClientStub(ProtocolHandler &protocol) : protocol_(protocol) {}

  // Opens a device, returning an opaque id with which to refer to it in
  // subsequent calls.
  using remote_device_id_t = uintptr_t;
  remote_device_id_t OpenDevice();

 private:
  ProtocolHandler &protocol_;
};

}  // namespace protocol_v1
}  // namespace remoting
}  // namespace iree

#endif  // IREE_REMOTING_PROTOCOL_V1_HAL_STUB_H_
