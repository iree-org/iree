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

// NOTE: unlike most VM modules we are only ever created and used from C++ code
// linked into the same library, because of this we can avoid the C shims and
// directly use C++ types.

#ifndef IREE_HAL_VMLA_VMLA_MODULE_H_
#define IREE_HAL_VMLA_VMLA_MODULE_H_

#include <cstdint>

#include "absl/types/span.h"
#include "iree/base/api.h"
#include "iree/base/memory.h"
#include "iree/base/ref_ptr.h"
#include "iree/base/status.h"
#include "iree/vm/api.h"
#include "iree/vm/module_abi_cc.h"
#include "iree/vm/types.h"

namespace iree {
namespace hal {
namespace vmla {

using iree_vmla_size_t = uint32_t;
using iree_vmla_shape_t = absl::Span<const int32_t>;

// Sentinel indicating that the remaining buffer after any offset has been
// applied should be used as the length.
constexpr iree_vmla_size_t kVMLAWholeBuffer = -1;

// A lightweight buffer lifetime management type.
// This is exported to modules as `vmla.buffer`. It can be used to provide
// views into existing rdata buffers (by specifying IREE_ALLOCATOR_NULL),
// views into parent buffers (parents retained via a reference), or dedicated
// allocations from an allocator.
//
// The provided data pointer and length is always for the buffer itself; it'll
// already be offset/clamped to parent buffer bounds when a view.
class Buffer final : public RefObject<Buffer> {
 public:
  static StatusOr<vm::ref<Buffer>> Allocate(size_t byte_length,
                                            iree_allocator_t allocator);

  static StatusOr<vm::ref<Buffer>> Wrap(const void* data, size_t data_length,
                                        iree_allocator_t allocator);

  static StatusOr<vm::ref<Buffer>> WrapMutable(void* data, size_t data_length,
                                               iree_allocator_t allocator);

  ~Buffer();

  constexpr const void* data() const { return data_; }
  constexpr void* data() { return data_; }
  constexpr size_t size() const { return data_length_; }

  template <typename T>
  absl::Span<const T> As() const {
    return absl::MakeConstSpan(reinterpret_cast<const T*>(data_),
                               data_length_ / sizeof(T));
  }

  template <typename T>
  absl::Span<T> As() {
    return absl::MakeSpan(reinterpret_cast<T*>(data_),
                          data_length_ / sizeof(T));
  }

  template <typename T>
  StatusOr<absl::Span<T>> RangeAs(iree_vmla_size_t byte_offset,
                                  iree_vmla_size_t byte_length) {
    ASSIGN_OR_RETURN(auto byte_range, MakeRange(byte_offset, byte_length));
    return ReinterpretSpan<T>(byte_range);
  }

 private:
  StatusOr<absl::Span<uint8_t>> MakeRange(iree_vmla_size_t byte_offset,
                                          iree_vmla_size_t byte_length) const;

  vm::ref<Buffer> parent_;
  void* data_ = nullptr;
  size_t data_length_ = 0;
  iree_allocator_t allocator_;
};

class Interface final : public RefObject<Interface> {
 public:
  static constexpr int kMaxConstants = 32;
  static constexpr int kMaxSets = 4;
  static constexpr int kMaxBindings = 32;

  struct Binding {
    vm::ref<Buffer> buffer;
    // TODO(benvanik): other descriptor set information.
  };

  // Resets all bindings on the interface.
  void Reset();

  // Gets the value from the push constants block at the given element offset.
  StatusOr<uint32_t> GetConstant(uint32_t offset) const;

  // Sets the push constant block contents to the given values.
  Status SetConstants(absl::Span<const uint32_t> values);

  // Gets the binding within a set. Note that the buffer may be null.
  StatusOr<const Binding> GetBinding(int32_t set, int32_t binding) const;

  // Sets a binding within a set to the given buffer value (possibly null).
  Status SetBinding(int32_t set, int32_t binding, Binding value);

 private:
  std::array<uint32_t, kMaxConstants> constants_;
  std::array<std::array<Binding, kMaxBindings>, kMaxSets> bindings_;
};

Status ModuleRegisterTypes();

Status ModuleCreate(iree_allocator_t allocator, iree_vm_module_t** out_module);

}  // namespace vmla
}  // namespace hal
}  // namespace iree

IREE_VM_DECLARE_TYPE_ADAPTERS(Buffer, iree::hal::vmla::Buffer);
IREE_VM_DECLARE_TYPE_ADAPTERS(Interface, iree::hal::vmla::Interface);

#endif  // IREE_HAL_VMLA_VMLA_MODULE_H_
