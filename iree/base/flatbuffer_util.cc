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

#include "iree/base/flatbuffer_util.h"

#include <cerrno>
#include <cstring>

#include "absl/memory/memory.h"
#include "iree/base/file_mapping.h"
#include "iree/base/memory.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"

namespace iree {

FlatBufferFileBase::~FlatBufferFileBase() {
  if (deleter_) {
    deleter_();
    deleter_ = []() {};
  }
}

Status FlatBufferFileBase::Create(const void* root_ptr,
                                  std::function<void()> deleter) {
  IREE_TRACE_SCOPE0("FlatBufferFileBase::Create");

  root_ptr_ = root_ptr;
  deleter_ = std::move(deleter);

  return OkStatus();
}

Status FlatBufferFileBase::CreateWithBackingBuffer(
    const void* root_ptr, ::flatbuffers::DetachedBuffer backing_buffer) {
  IREE_TRACE_SCOPE0("FlatBufferFileBase::Create");

  root_ptr_ = root_ptr;

  // Pass along the buffer provided so we keep it alive until the
  // FlatBufferFileBase is destructed.
  auto backing_buffer_baton = IreeMoveToLambda(backing_buffer);
  deleter_ = [backing_buffer_baton]() { (void)backing_buffer_baton.value; };

  return OkStatus();
}

Status FlatBufferFileBase::Wrap(const void* root_ptr) {
  IREE_TRACE_SCOPE0("FlatBufferFileBase::Wrap");
  return Create(root_ptr, []() {});
}

Status FlatBufferFileBase::FromBuffer(Identifier identifier,
                                      absl::Span<const uint8_t> buffer_data,
                                      std::function<void()> deleter,
                                      size_t root_type_size,
                                      VerifierFn verifier_fn) {
  IREE_TRACE_SCOPE("FlatBufferFileBase::FromBuffer:size", int)
  (static_cast<int>(buffer_data.size()));

  // Sanity check buffer for the minimum size as FlatBuffers doesn't.
  if (buffer_data.size() < 16) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Provided serialized flatbuffer buffer is too small to be legit "
              "at size="
           << buffer_data.size();
  }

  // Ensure the buffer has the BIPE magic bytes.
  if (identifier.has_value() && !::flatbuffers::BufferHasIdentifier(
                                    buffer_data.data(), identifier.value())) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Provided serialized buffer does not contain the expected type; "
              "magic bytes mismatch (expected "
           << identifier.value() << ")";
  }

  // Verify the FlatBuffer contains valid offsets and won't try to read out of
  // bounds of the buffer. We inline a bit of VerifyBufferFromStart so this code
  // can stay generic.
  {
    IREE_TRACE_SCOPE0("FlatBufferFileBase::FromBufferVerification");
    ::flatbuffers::Verifier verifier{buffer_data.data(), buffer_data.size()};
    if (!verifier_fn(identifier.value_or(nullptr), &verifier)) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "FlatBuffer failed to verify as expected type; possibly "
                "corrupt input";
    }
  }

  // Resolve the root pointer in the buffer.
  // This is GetMutableRoot such that we don't need to know T.
  root_ptr_ = buffer_data.data() +
              ::flatbuffers::EndianScalar(
                  *reinterpret_cast<const ::flatbuffers::uoffset_t*>(
                      buffer_data.data()));
  if (!root_ptr_) {
    return FailedPreconditionErrorBuilder(IREE_LOC)
           << "Unable to resolve root table";
  }
  deleter_ = std::move(deleter);

  return OkStatus();
}

Status FlatBufferFileBase::WrapBuffer(Identifier identifier,
                                      absl::Span<const uint8_t> buffer_data,
                                      size_t root_type_size,
                                      VerifierFn verifier_fn) {
  IREE_TRACE_SCOPE0("FlatBufferFileBase::WrapBuffer");
  return FromBuffer(
      identifier, buffer_data, []() {}, root_type_size, verifier_fn);
}

Status FlatBufferFileBase::LoadFile(Identifier identifier, std::string path,
                                    size_t root_type_size,
                                    VerifierFn verifier_fn) {
  IREE_TRACE_SCOPE0("FlatBufferFileBase::LoadFile");

  IREE_ASSIGN_OR_RETURN(auto file_mapping, FileMapping::OpenRead(path));
  auto buffer_data = file_mapping->data();

  auto handle_baton = IreeMoveToLambda(file_mapping);
  return FromBuffer(
      identifier, buffer_data,
      [handle_baton]() {
        // Keeping the mmap handle alive.
        (void)handle_baton.value;
      },
      root_type_size, verifier_fn);
}

}  // namespace iree
