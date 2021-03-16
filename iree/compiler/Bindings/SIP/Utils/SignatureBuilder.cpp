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

#include "iree/compiler/Bindings/SIP/Utils/SignatureBuilder.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace SIP {

// -----------------------------------------------------------------------------
// SignatureBuilder
// -----------------------------------------------------------------------------

SignatureBuilder& SignatureBuilder::Integer(int value, char tag) {
  assert(tag == '_' || (tag >= 'a' && tag <= 'z') &&
                           "integer signature tag must be '_' or 'a'..'z'");
  encoded_.push_back(tag);
  encoded_.append(std::to_string(value));
  return *this;
}

SignatureBuilder& SignatureBuilder::Span(StringRef contents, char tag) {
  assert((tag >= 'A' && tag <= 'Z') && "span signature tag must be 'A'..'Z'");
  encoded_.push_back(tag);
  // If the contents starts with a digit or the escape char (!), then escape it.
  encoded_.append(std::to_string(contents.size() + 1));
  encoded_.push_back('!');
  encoded_.append(contents.begin(), contents.end());
  return *this;
}

// -----------------------------------------------------------------------------
// RawSignatureMangler
// -----------------------------------------------------------------------------

SignatureBuilder RawSignatureMangler::ToFunctionSignature(
    const SignatureBuilder& inputs, const SignatureBuilder& results) {
  SignatureBuilder func_builder;
  inputs.AppendTo(func_builder, 'I');
  results.AppendTo(func_builder, 'R');
  return func_builder;
}

void RawSignatureMangler::AddUnrecognized() { builder_.Span(StringRef(), 'U'); }

void RawSignatureMangler::AddAnyReference() {
  // A more constrained ref object would have a non empty span.
  builder_.Span(StringRef(), 'O');
}

void RawSignatureMangler::AddShapedNDBuffer(
    AbiConstants::ScalarType element_type, ArrayRef<int> shape) {
  SignatureBuilder item_builder;
  // Fields:
  //   't': scalar type code
  //   'd': shape dimension
  if (static_cast<unsigned>(element_type) != 0) {
    item_builder.Integer(static_cast<unsigned>(element_type), 't');
  }
  for (int d : shape) {
    item_builder.Integer(d, 'd');
  }
  item_builder.AppendTo(builder_, 'B');
}

void RawSignatureMangler::AddScalar(AbiConstants::ScalarType type) {
  SignatureBuilder item_builder;
  // Fields:
  //   't': scalar type code
  if (static_cast<unsigned>(type) != 0) {
    item_builder.Integer(static_cast<unsigned>(type), 't');
  }
  item_builder.AppendTo(builder_, 'S');
}

// -----------------------------------------------------------------------------
// SipSignatureMangler
// -----------------------------------------------------------------------------

SipSignatureMangler::SipSignatureMangler() = default;

bool SipSignatureMangler::SetRawSignatureIndex(int raw_signature_index,
                                               ArrayRef<Key> path) {
  if (raw_signature_index < 0) {
    return false;
  }

  Value* level = &root_;
  for (const auto& key : path) {
    // Is the indexing mode compatible?
    if (level->index_mode == IndexMode::kNone) {
      // Not yet committed: just adopt this first access.
      level->index_mode = key.index_mode();
    } else if (level->index_mode != key.index_mode()) {
      // Indexing mode mismatch.
      return false;
    }

    auto found_it = level->children.find(key);
    if (found_it == level->children.end()) {
      // Create a new level.
      auto child = std::make_unique<Value>();
      Value* unowned_child = child.get();
      level->children.insert(std::make_pair(key, std::move(child)));
      level = unowned_child;
      continue;
    }

    // Found.
    level = found_it->second.get();
  }

  // Should now be on the leaf/terminal.
  if (level->index_mode != IndexMode::kNone ||
      level->raw_signature_index != -1) {
    // It is not a leaf or has already been setup as a leaf.
    return false;
  }

  level->raw_signature_index = raw_signature_index;
  return true;
}

bool SipSignatureMangler::ToStructureSignature(SignatureBuilder* sb,
                                               const Value* level) const {
  char sub_span_tag;
  switch (level->index_mode) {
    case IndexMode::kNone:
      // Leaf with un-assigned raw index.
      if (level->raw_signature_index < 0) {
        // An un-assigned leaf is only allowed for the root.
        assert(level == &root_ && "Un-assigned non-root leaf not allowed");
        return level == &root_;
      } else {
        sb->Integer(level->raw_signature_index);
        return true;
      }
    case IndexMode::kSequence:
      sub_span_tag = 'S';
      break;
    case IndexMode::kDict:
      sub_span_tag = 'D';
      break;
    default:
      return false;
  }

  SignatureBuilder child_sb;
  for (const auto& kv : level->children) {
    const Key& key = kv.first;
    if (key.is_integer_key()) {
      child_sb.Integer(key.ikey(), 'k');
    } else if (key.is_string_key()) {
      child_sb.Span(key.skey(), 'K');
    } else {
      return false;
    }
    if (!ToStructureSignature(&child_sb, kv.second.get())) return false;
  }

  child_sb.AppendTo(*sb, sub_span_tag);
  return true;
}

llvm::Optional<SignatureBuilder> SipSignatureMangler::ToFunctionSignature(
    const SipSignatureMangler& inputs_struct,
    const SipSignatureMangler& results_struct) {
  auto inputs_sb = inputs_struct.ToStructureSignature();
  auto results_sb = results_struct.ToStructureSignature();

  if (!inputs_sb || !results_sb) return {};

  SignatureBuilder func_sb;
  inputs_sb->AppendTo(func_sb, 'I');
  results_sb->AppendTo(func_sb, 'R');
  return func_sb;
}

}  // namespace SIP
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
