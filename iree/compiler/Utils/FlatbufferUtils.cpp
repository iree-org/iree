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

#include "iree/compiler/Utils/FlatbufferUtils.h"

#include <vector>

#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace iree_compiler {

FlatbufferBuilder::FlatbufferBuilder() { flatcc_builder_init(&builder); }

FlatbufferBuilder::~FlatbufferBuilder() { flatcc_builder_clear(&builder); }

flatbuffers_uint8_vec_ref_t FlatbufferBuilder::streamUint8Vec(
    std::function<bool(raw_ostream &stream)> fn) {
  flatbuffers_uint8_vec_start(*this);
  raw_flatbuffer_uint8_vec_ostream stream(*this);
  if (!fn(stream)) {
    return 0;
  }
  stream.flush();
  return flatbuffers_uint8_vec_end(*this);
}

DenseIntElementsAttr FlatbufferBuilder::getBufferAttr(MLIRContext *context) {
  // NOTE: this is a alloc/copy. We need to have a single contiguous buffer to
  // pass into the elements factory function and the data we have in the
  // builder is paged. If we end up with a custom attribute type for this that
  // does not support storage uniquing then we can directly allocate and copy
  // the pages into the buffer without the extra copy.
  size_t packedSize = flatcc_builder_get_buffer_size(*this);
  std::vector<uint8_t> packedData(packedSize);
  void *result =
      flatcc_builder_copy_buffer(*this, packedData.data(), packedData.size());
  assert(result && "flatcc_emitter_t impl failed (non-default?)");

  // NOTE: ew. OpaqueAttr may be better? It does equality checks but won't try
  // to unique and would let us get a mutable buffer out.
  return DenseIntElementsAttr::get(
      VectorType::get({static_cast<int64_t>(packedSize)},
                      IntegerType::get(8, context)),
      std::move(packedData));
}

}  // namespace iree_compiler
}  // namespace mlir
