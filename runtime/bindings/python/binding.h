// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BINDINGS_PYTHON_IREE_BINDING_H_
#define IREE_BINDINGS_PYTHON_IREE_BINDING_H_

#include <optional>
#include <vector>

#include "iree/base/api.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace iree {
namespace python {

namespace py = pybind11;

template <typename T>
struct ApiPtrAdapter {};

template <typename Self, typename T>
class ApiRefCounted {
 public:
  using RawPtrType = T*;
  ApiRefCounted() : instance_(nullptr) {}
  ApiRefCounted(ApiRefCounted& other) : instance_(other.instance_) { Retain(); }
  ApiRefCounted(ApiRefCounted&& other) : instance_(other.instance_) {
    other.instance_ = nullptr;
  }
  ApiRefCounted& operator=(ApiRefCounted&& other) {
    instance_ = other.instance_;
    other.instance_ = nullptr;
    return *this;
  }
  void operator=(const ApiRefCounted&) = delete;

  ~ApiRefCounted() { Release(); }

  // Steals the reference to the object referenced by the given raw pointer and
  // returns a wrapper (transfers ownership).
  static Self StealFromRawPtr(T* retained_inst) {
    auto self = Self();
    self.instance_ = retained_inst;
    return self;
  }

  // Retains the object referenced by the given raw pointer and returns
  // a wrapper.
  static Self BorrowFromRawPtr(T* non_retained_inst) {
    auto self = Self();
    self.instance_ = non_retained_inst;
    if (non_retained_inst) {
      ApiPtrAdapter<T>::Retain(non_retained_inst);
    }
    return self;
  }

  // Whether it is nullptr.
  operator bool() const { return instance_; }

  T* steal_raw_ptr() {
    T* ret = instance_;
    instance_ = nullptr;
    return ret;
  }

  T* raw_ptr() {
    if (!instance_) {
      throw std::invalid_argument("API object is null");
    }
    return instance_;
  }

  const T* raw_ptr() const {
    return const_cast<ApiRefCounted*>(this)->raw_ptr();
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

#endif  // IREE_BINDINGS_PYTHON_IREE_BINDING_H_
