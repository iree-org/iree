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

#include "third_party/mlir_edge/iree/base/flatbuffer_util.h"

#include <cerrno>
#include <cstring>

#include "third_party/absl/memory/memory.h"
#include "third_party/absl/types/source_location.h"
#include "third_party/mlir_edge/iree/base/memory.h"
#include "third_party/mlir_edge/iree/base/status.h"
#include "third_party/mlir_edge/iree/base/tracing.h"

// Used for mmap:
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

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
  auto backing_buffer_baton = MoveToLambda(backing_buffer);
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
    return InvalidArgumentErrorBuilder(ABSL_LOC)
           << "Provided serialized flatbuffer buffer is too small to be legit "
              "at size="
           << buffer_data.size();
  }

  // Ensure the buffer has the BIPE magic bytes.
  if (identifier.has_value() && !::flatbuffers::BufferHasIdentifier(
                                    buffer_data.data(), identifier.value())) {
    return InvalidArgumentErrorBuilder(ABSL_LOC)
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
      return InvalidArgumentErrorBuilder(ABSL_LOC)
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
    return FailedPreconditionErrorBuilder(ABSL_LOC)
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

Status FlatBufferFileBase::FromString(Identifier identifier,
                                      std::string buffer_data,
                                      size_t root_type_size,
                                      VerifierFn verifier_fn) {
  IREE_TRACE_SCOPE0("FlatBufferFileBase::FromString");

  // Reference right into the string buffer.
  auto buffer_data_data = absl::MakeConstSpan(
      reinterpret_cast<const uint8_t*>(buffer_data.data()), buffer_data.size());

  // Use a baton to keep the string alive until the FlatBufferFileBase is
  // destroyed.
  auto buffer_data_baton = MoveToLambda(buffer_data);
  return FromBuffer(
      identifier, buffer_data_data,
      [buffer_data_baton]() {
        // Keeping the string alive.
        (void)buffer_data_baton.value;
      },
      root_type_size, verifier_fn);
}

namespace {

class FileDescriptor {
 public:
  static StatusOr<std::unique_ptr<FileDescriptor> > OpenRead(std::string path) {
    struct stat buf;
    if (::lstat(path.c_str(), &buf) == -1) {
      return NotFoundErrorBuilder(ABSL_LOC)
             << "Unable to stat file " << path << ": " << ::strerror(errno);
    }

    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd == -1) {
      return UnavailableErrorBuilder(ABSL_LOC)
             << "Unable to open file " << path << ": " << ::strerror(errno);
    }
    return absl::WrapUnique(
        new FileDescriptor(fd, static_cast<size_t>(buf.st_size)));
  }

  ~FileDescriptor() { ::close(fd_); }

  int fd() const { return fd_; }
  size_t size() const { return size_; }

 private:
  FileDescriptor(int fd, size_t size) : fd_(fd), size_(size) {}
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;

  int fd_;
  size_t size_;
};

class MappedFile {
 public:
  static StatusOr<std::unique_ptr<MappedFile> > Open(absl::string_view path) {
    // Open the file for reading. Note that we only need to keep it open long
    // enough to map it and we can close the descriptor after that.
    ASSIGN_OR_RETURN(auto file, FileDescriptor::OpenRead(std::string(path)));

    // Map the file from the file descriptor.
    void* data =
        ::mmap(nullptr, file->size(), PROT_READ, MAP_SHARED, file->fd(), 0);
    if (data == MAP_FAILED) {
      return UnavailableErrorBuilder(ABSL_LOC)
             << "Mapping failed on file (ensure uncompressed): " << path;
    }

    return absl::WrapUnique(new MappedFile(data, file->size()));
  }

  ~MappedFile() {
    if (::munmap(const_cast<void*>(data_), data_size_) != 0) {
      LOG(WARNING) << "Unable to unmap file: " << strerror(errno);
    }
  }

  const void* data() const { return data_; }
  size_t data_size() const { return data_size_; }

 private:
  MappedFile(const void* data, size_t data_size)
      : data_(data), data_size_(data_size) {}
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  const void* data_;
  size_t data_size_;
};

}  // namespace

Status FlatBufferFileBase::LoadFile(Identifier identifier,
                                    absl::string_view path,
                                    size_t root_type_size,
                                    VerifierFn verifier_fn) {
  IREE_TRACE_SCOPE0("FlatBufferFileBase::LoadFile");

  ASSIGN_OR_RETURN(auto mapped_file, MappedFile::Open(path));

  absl::Span<const uint8_t> buffer_data{
      reinterpret_cast<const uint8_t*>(mapped_file->data()),
      mapped_file->data_size()};

  auto handle_baton = MoveToLambda(mapped_file);
  return FromBuffer(
      identifier, buffer_data,
      [handle_baton]() {
        // Keeping the mmap handle alive.
        (void)handle_baton.value;
      },
      root_type_size, verifier_fn);
}

}  // namespace iree
