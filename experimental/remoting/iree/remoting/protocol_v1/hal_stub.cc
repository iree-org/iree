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

#include "experimental/remoting/iree/remoting/protocol_v1/hal_stub.h"

#include "experimental/remoting/iree/remoting/schemas/protocol_v1_builder.h"

namespace iree {
namespace remoting {
namespace protocol_v1 {

#undef ns
#define ns(x) FLATBUFFERS_WRAP_NAMESPACE(iree_remoting_protocol_v1, x)

HalClientStub::remote_device_id_t HalClientStub::OpenDevice() {
  remote_device_id_t remote_device_id = 1;  // TODO
  flatcc_builder_t *B = protocol_.StartPacket();
  uint32_t correlation_id = 1;  // TODO

  // Create HalDeviceOpenRequest.
  ns(HalDeviceOpenRequest_ref_t) request =
      ns(HalDeviceOpenRequest_create(B, remote_device_id));
  ns(AnyRequest_union_ref_t) request_union =
      ns(AnyRequest_as_HalDeviceOpenRequest(request));

  ns(Envelope_start_as_root(B));
  ns(Envelope_request_add(B, request_union));
  ns(Envelope_correlation_id_add(B, correlation_id));
  ns(Envelope_end_as_root(B));

  protocol_.SendPacket();
  return remote_device_id;
}

}  // namespace protocol_v1
}  // namespace remoting
}  // namespace iree
