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

#ifndef IREE_COMPILER_UTILS_FLATBUFFERUTILS_H_
#define IREE_COMPILER_UTILS_FLATBUFFERUTILS_H_

#include <functional>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

extern "C" {
// NOTE: order matters here (common_reader must preceed common_builder).
// clang-format off
#include "flatcc/flatcc_builder.h"
#include "flatcc/reflection/flatbuffers_common_reader.h"
#include "flatcc/reflection/flatbuffers_common_builder.h"
// clang-format on
}  // extern "C"

namespace mlir {
namespace iree_compiler {

// RAII wrapper for flatcc_builder_t; pass to functions requiring a builder.
//
// Usage:
//   FlatbufferBuilder builder;
//   // NOTE: flatbuffers are built bottoms-up so we first generate our [uint8]:
//   auto dataRef = builder.streamUint8Vec(...);
//   // ... and then start the table that references it:
//   my_type_start_as_root(builder);
//   my_type_uint8_vec_field_add(builder, dataRef);
//   my_type_end_as_root(builder);
//   // ... and finally capture the results as an mlir::Attribute.
//   auto attr = builder.getBufferAttr(mlirContext);
class FlatbufferBuilder {
 public:
  FlatbufferBuilder();
  ~FlatbufferBuilder();

  operator flatcc_builder_t *() { return &builder; }

  // Creates a string vector containing all strings in the given range.
  template <typename RangeTy>
  flatbuffers_string_vec_ref_t createStringVec(RangeTy &&Range) {
    auto stringRefs =
        llvm::to_vector<8>(llvm::map_range(Range, [&](StringRef value) {
          return flatbuffers_string_create(*this, value.data(), value.size());
        }));
    return flatbuffers_string_vec_create(*this, stringRefs.data(),
                                         stringRefs.size());
  }

  // Provides a raw_ostream that |fn| can use to directly stream into a [uint8]
  // in the flatbuffer builder.
  //
  // Usage:
  //   auto ref = builder.streamUint8Vec([&](llvm::raw_ostream &stream) {
  //     stream << "foo";
  //     return true;
  //   });
  //   ...
  //   my_type_uint8_vec_field_add(builder, ref);  // use vec reference
  //   ...
  flatbuffers_uint8_vec_ref_t streamUint8Vec(
      std::function<bool(raw_ostream &stream)> fn);

  // Captures the current contents of the flatbuffer builder and returns them
  // as a shaped `vector<SIZExi8>` dense attr. The builder is left unmodified.
  DenseIntElementsAttr getBufferAttr(MLIRContext *context);

 private:
  flatcc_builder_t builder;
};

// Allows streaming bytes directly into a flatbuffer `[uint8]` field.
// The ostream runs in buffered mode and routes all writes into pages
// allocated by the flatbuffer builder as we grow the output.
//
// Usage:
//   flatbuffers_uint8_vec_start(builder);
//   raw_flatbuffer_uint8_vec_ostream stream(builder);
//   stream << "foo";
//   stream.flush();  // *********** IMPORTANT ***********
//   flatbuffers_uint8_vec_ref_t ref = flatbuffers_uint8_vec_end(builder);
class raw_flatbuffer_uint8_vec_ostream : public llvm::raw_ostream {
 public:
  explicit raw_flatbuffer_uint8_vec_ostream(flatcc_builder_t *builder)
      : raw_ostream(/*unbuffered=*/true), builder(builder) {}

  ~raw_flatbuffer_uint8_vec_ostream() override { flush(); }

 private:
  void write_impl(const char *Ptr, size_t Size) override {
    flatbuffers_uint8_vec_append(builder,
                                 reinterpret_cast<const uint8_t *>(Ptr), Size);
  }

  uint64_t current_pos() const override {
    return tell() - GetNumBytesInBuffer();
  }

  flatcc_builder_t *builder;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_UTILS_FLATBUFFERUTILS_H_
