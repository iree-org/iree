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

#ifndef IREE_BINDINGS_PYTHON_PYIREE_FUNCTION_ABI_H_
#define IREE_BINDINGS_PYTHON_PYIREE_FUNCTION_ABI_H_

#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "bindings/python/pyiree/binding.h"
#include "bindings/python/pyiree/hal.h"
#include "bindings/python/pyiree/host_types.h"
#include "iree/base/signature_mangle.h"

namespace iree {
namespace python {

// Forward declarations.
class RtContext;

// A HalBuffer (iree_hal_buffer_t) bound to a function argument.
// At this point, the buffer has been completely validated, with all shape
// information erased except for any dynamic dims.
struct BoundHalBufferFunctionArg {
  // The backing buffer.
  HalBuffer buffer;
  // If this function argument is backed by a python object, it is retained
  // here.
  py::object dependent_pyobject;
  // Dynamic dims in the shape (for shaped buffers).
  absl::InlinedVector<int, 2> dynamic_dims;
};

// Opaque (to python) native argument.
using FunctionArgVariant =
    absl::variant<std::nullptr_t, BoundHalBufferFunctionArg>;

// Opaque list of function arguments.
// Has sufficient accessors on it to facilitate printing and testing but is
// otherwise, not visible to python.
// Typically, native code will interact with the lower level span based API
// directly (and avoid some overhead). Therefore, this class does not seek to
// be optimal.
class FunctionArgVariantList {
 public:
  using VectorType = absl::InlinedVector<FunctionArgVariant, 4>;
  FunctionArgVariantList() = default;
  FunctionArgVariantList(VectorType contents)
      : contents_(std::move(contents)) {}

  VectorType& contents() { return contents_; }
  const VectorType& contents() const { return contents_; }

 private:
  VectorType contents_;
};

// Instantiated with function attributes in order to process inputs/outputs.
class FunctionAbi {
 public:
  using AttributeLookup =
      std::function<absl::optional<absl::string_view>(absl::string_view)>;
  FunctionAbi(std::shared_ptr<HostTypeFactory> host_type_factory)
      : host_type_factory_(std::move(host_type_factory)) {}
  virtual ~FunctionAbi() = default;

  using Description = RawSignatureParser::Description;
  using InputDescriptionVector = absl::InlinedVector<Description, 4>;
  using ResultDescriptionVector = absl::InlinedVector<Description, 1>;

  struct RawConfig {
    InputDescriptionVector inputs;
    ResultDescriptionVector results;

    // The following are retained to aid debugging but may be empty if
    // disabled.
    std::string signature;
  };

  // Creates an instance based on the function attributes.
  static std::unique_ptr<FunctionAbi> Create(
      std::shared_ptr<HostTypeFactory> host_type_factory,
      AttributeLookup lookup);

  RawConfig& raw_config() { return raw_config_; }
  int raw_input_arity() const { return raw_config_.inputs.size(); }
  int raw_result_arity() const { return raw_config_.results.size(); }

  // Raw packing and unpacking. These always operate on the linear span
  // of raw inputs and results. Some ABIs perform a higher level of mapping
  // on top of this, which can be accessed via the non-prefixed Pack/Unpack
  // methods.
  // Given a span of descriptions, packs the given py_args into the span
  // of function args. All spans must be of the same size.
  void RawPack(RtContext& context, absl::Span<const Description> descs,
               absl::Span<py::handle> py_args,
               absl::Span<FunctionArgVariant> f_args, bool writable);

  // Given bound function arguments (from RawPack or equiv) and signature
  // descriptors, allocates results for the function invocation. For fully
  // specified result types, this can be done purely by matching up
  // reflection metadata and an oracle for determining layout. For dynamically
  // shaped or data-dependent shaped results, the metadata about the function
  // arguments may be required to generate additional allocation function calls.
  // Finally, in truly data-dependent cases, some results may not be resolvable
  // ahead of time, resulting in a nullptr in f_results. In such cases, the
  // invocation must ensure proper barriers are in place to fully execute the
  // function prior to delivering results to the user layer.
  void AllocateResults(RtContext& context, absl::Span<const Description> descs,
                       absl::Span<const FunctionArgVariant> f_args,
                       absl::Span<FunctionArgVariant> f_results);

  // Gets the string representation.
  std::string DebugString() const;

 private:
  std::shared_ptr<HostTypeFactory> host_type_factory_;
  RawConfig raw_config_;
};

void SetupFunctionAbiBindings(pybind11::module m);

}  // namespace python
}  // namespace iree

#endif  // IREE_BINDINGS_PYTHON_PYIREE_FUNCTION_ABI_H_
