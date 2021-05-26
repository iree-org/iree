// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BINDINGS_PYTHON_IREE_RT_HOST_TYPES_H_
#define IREE_BINDINGS_PYTHON_IREE_RT_HOST_TYPES_H_

#include <array>

#include "absl/types/span.h"
#include "bindings/python/iree/runtime/binding.h"
#include "bindings/python/iree/runtime/hal.h"
#include "iree/base/signature_parser.h"

namespace iree {
namespace python {

extern const std::array<
    const char*,
    static_cast<unsigned>(AbiConstants::ScalarType::kMaxScalarType) + 1>
    kScalarTypePyFormat;
extern const std::array<
    uint32_t,
    static_cast<unsigned>(AbiConstants::ScalarType::kMaxScalarType) + 1>
    kScalarTypeToHalElementType;

class HostTypeFactory {
 public:
  virtual ~HostTypeFactory() = default;

  // Creates a default implementation which interops with numpy.
  static std::shared_ptr<HostTypeFactory> GetNumpyFactory();

  // Creates a C-contiguous ndarray of the given element_type/dims and backed
  // by the given buffer. The resulting array has no synchronization and is
  // available for use immediately.
  virtual py::object CreateImmediateNdarray(
      AbiConstants::ScalarType element_type, absl::Span<const int> dims,
      HalBuffer buffer, py::object parent_keep_alive);

  // TODO(laurenzo): Add a CreateDelayedNdarray() which is conditioned on
  // a semaphore. This is actually what should be used for async results.
};

void SetupHostTypesBindings(pybind11::module m);

}  // namespace python
}  // namespace iree

#endif  // IREE_BINDINGS_PYTHON_IREE_RT_HOST_TYPES_H_
