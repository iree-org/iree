// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_TESTING_SCOPED_RESOURCE_H_
#define IREE_TOKENIZER_TESTING_SCOPED_RESOURCE_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::tokenizer::testing {

//===----------------------------------------------------------------------===//
// ScopedResource
//===----------------------------------------------------------------------===//

// Move-only RAII wrapper for C resources freed via a function pointer.
//
// Usage:
//   using ScopedModel = ScopedResource<
//       iree_tokenizer_model_t, iree_tokenizer_model_free>;
//   ScopedModel model(raw_model_ptr);
//   model->some_field;           // pointer access
//   func(model.get());           // raw pointer
//   auto* ptr = model.release(); // transfer ownership
//   allocate_fn(model.put());    // populate via out-param
template <typename T, void (*FreeFn)(T*)>
class ScopedResource {
 public:
  ScopedResource() = default;
  explicit ScopedResource(T* pointer) : pointer_(pointer) {}
  ~ScopedResource() { FreeFn(pointer_); }

  ScopedResource(ScopedResource&& other) noexcept
      : pointer_(std::exchange(other.pointer_, nullptr)) {}
  ScopedResource& operator=(ScopedResource&& other) noexcept {
    if (this != &other) {
      FreeFn(pointer_);
      pointer_ = std::exchange(other.pointer_, nullptr);
    }
    return *this;
  }

  ScopedResource(const ScopedResource&) = delete;
  ScopedResource& operator=(const ScopedResource&) = delete;

  T* get() const { return pointer_; }
  T* operator->() const { return pointer_; }
  explicit operator bool() const { return pointer_ != nullptr; }

  T* release() { return std::exchange(pointer_, nullptr); }

  // Returns address of internal pointer for use with _allocate functions.
  // Releases any existing resource first to prevent leaks.
  T** put() {
    FreeFn(pointer_);
    pointer_ = nullptr;
    return &pointer_;
  }

 private:
  T* pointer_ = nullptr;
};

//===----------------------------------------------------------------------===//
// ScopedState
//===----------------------------------------------------------------------===//

// RAII wrapper for streaming state with automatic storage allocation.
// All tokenizer components follow the same state lifecycle:
//   size = component_state_size(component)
//   storage = allocate(size)
//   component_state_initialize(component, storage, &state)
//   // ... use state ...
//   component_state_deinitialize(state)
//
// Usage:
//   using ScopedModelState = ScopedState<
//       iree_tokenizer_model_t, iree_tokenizer_model_state_t,
//       iree_tokenizer_model_state_size,
//       iree_tokenizer_model_state_initialize,
//       iree_tokenizer_model_state_deinitialize>;
//   ScopedModelState state(model.get());
//   state.get();    // raw state pointer
//   state.Reset();  // reinitialize without reallocating
template <typename ComponentT, typename StateT,
          iree_host_size_t (*SizeFn)(const ComponentT*),
          iree_status_t (*InitFn)(const ComponentT*, void*, StateT**),
          void (*DeinitFn)(StateT*)>
class ScopedState {
 public:
  explicit ScopedState(ComponentT* component) : component_(component) {
    IREE_ASSERT_ARGUMENT(component);
    iree_host_size_t state_size = SizeFn(component);
    storage_.resize(state_size);
    IREE_CHECK_OK(InitFn(component, storage_.data(), &state_));
  }

  ~ScopedState() {
    if (state_) {
      DeinitFn(state_);
    }
  }

  ScopedState(const ScopedState&) = delete;
  ScopedState& operator=(const ScopedState&) = delete;
  ScopedState(ScopedState&&) = delete;
  ScopedState& operator=(ScopedState&&) = delete;

  StateT* get() { return state_; }

  // Reinitializes state without reallocating storage.
  void Reset() {
    if (state_) {
      DeinitFn(state_);
    }
    IREE_CHECK_OK(InitFn(component_, storage_.data(), &state_));
  }

 private:
  ComponentT* component_;
  std::vector<uint8_t> storage_;
  StateT* state_ = nullptr;
};

}  // namespace iree::tokenizer::testing

#endif  // IREE_TOKENIZER_TESTING_SCOPED_RESOURCE_H_
