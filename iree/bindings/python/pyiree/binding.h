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

#ifndef IREE_BINDINGS_PYTHON_PYIREE_BINDING_H_
#define IREE_BINDINGS_PYTHON_PYIREE_BINDING_H_

#include <vector>

#include "absl/types/optional.h"
#include "iree/base/api.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace pybind11 {
namespace detail {
#if !defined(ABSL_HAVE_STD_OPTIONAL)
// Make absl::optional act like the future C++17 optional for pybind11.
// If ABSL_HAVE_STD_OPTIONAL is defined then absl::optional == std::optional
// and the default type caster is sufficient.
template <typename T>
struct type_caster<absl::optional<T>> : optional_caster<absl::optional<T>> {};
#endif
}  // namespace detail
}  // namespace pybind11

namespace iree {
namespace python {

namespace py = pybind11;

// Wrapper around a blob of memory.
// Used to transport blobs back and forth between C++ and Python.
class OpaqueBlob {
 public:
  OpaqueBlob() : data_(nullptr), size_(0) {}
  OpaqueBlob(void* data, size_t size) : data_(data), size_(size) {}
  virtual ~OpaqueBlob() = default;

  void* data() { return data_; }
  const void* data() const { return data_; }
  size_t size() const { return size_; }

  // Create a free function from the OpaqueBlob shared pointer.
  using BufferFreeFn = void (*)(void* self, iree_byte_span_t);
  static std::pair<BufferFreeFn, void*> CreateFreeFn(
      std::shared_ptr<OpaqueBlob> blob) {
    // Note that there are more efficient ways to write this which
    // don't bounce through an extra heap alloc, but this is not
    // intended to be a high impact code path.
    struct Holder {
      std::shared_ptr<OpaqueBlob> blob;
    };
    Holder* holder = new Holder{std::move(blob)};
    auto free_fn = +([](void* self, iree_byte_span_t) {
      Holder* self_holder = static_cast<Holder*>(self);
      delete self_holder;
    });
    return {free_fn, holder};
  }

 protected:
  void* data_;
  size_t size_;
};

// Opaque blob that owns a vector.
class OpaqueByteVectorBlob : public OpaqueBlob {
 public:
  OpaqueByteVectorBlob(std::vector<uint8_t> v)
      : OpaqueBlob(), v_(std::move(v)) {
    data_ = v_.data();
    size_ = v_.size();
  }

 private:
  std::vector<uint8_t> v_;
};

template <typename T>
struct ApiPtrAdapter {};

template <typename Self, typename T>
class ApiRefCounted {
 public:
  ApiRefCounted() : instance_(nullptr) {}
  ApiRefCounted(ApiRefCounted&& other) : instance_(other.instance_) {
    other.instance_ = nullptr;
  }
  void operator=(const ApiRefCounted&) = delete;

  ~ApiRefCounted() { Release(); }

  // Creates an instance of the ref counted wrapper based on an instance
  // that has already been retained. Ownership is transferred to the
  // wrapper.
  static Self CreateRetained(T* retained_inst) {
    auto self = Self();
    self.instance_ = retained_inst;
    return self;
  }

  // Creates a new instance, retaining the underlying object.
  static Self RetainAndCreate(T* non_retained_inst) {
    auto self = Self();
    self.instance_ = non_retained_inst;
    if (non_retained_inst) {
      ApiPtrAdapter<T>::Retain(non_retained_inst);
    }
    return self;
  }

  T* raw_ptr() {
    if (!instance_) {
      throw std::invalid_argument("API object is null");
    }
    return instance_;
  }
  void Retain() {
    if (instance_) {
      ApiPtrAdapter<T>::Retain(instance_);
    }
  }
  void Release() {
    if (instance_) {
      ApiPtrAdapter<T>::Release(instance_);
    }
  }

 private:
  T* instance_;
};

}  // namespace python
}  // namespace iree

#endif  // IREE_BINDINGS_PYTHON_PYIREE_BINDING_H_
