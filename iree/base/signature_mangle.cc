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

#include "iree/base/signature_mangle.h"

#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"

namespace iree {

// -----------------------------------------------------------------------------
// AbiConstants
// -----------------------------------------------------------------------------

const std::array<size_t, 12> AbiConstants::kScalarTypeSize = {
    4,  // kIeeeFloat32 = 0,
    2,  // kIeeeFloat16 = 1,
    8,  // kIeeeFloat64 = 2,
    2,  // kGoogleBfloat16 = 3,
    1,  // kSint8 = 4,
    2,  // kSint16 = 5,
    4,  // kSint32 = 6,
    8,  // kSint64 = 7,
    1,  // kUint8 = 8,
    2,  // kUint16 = 9,
    4,  // kUint32 = 10,
    8,  // kUint64 = 11,
};

const std::array<const char*, 12> AbiConstants::kScalarTypeNames = {
    "float32", "float16", "float64", "bfloat16", "sint8",  "sint16",
    "sint32",  "sint64",  "uint8",   "uint16",   "uint32", "uint64",
};

// -----------------------------------------------------------------------------
// SignatureBuilder and SignatureParser
// -----------------------------------------------------------------------------

SignatureBuilder& SignatureBuilder::Integer(int value, char tag) {
  assert(tag == '_' || (tag >= 'a' && tag <= 'z') &&
                           "integer signature tag must be '_' or 'a'..'z'");
  encoded_.push_back(tag);
  absl::StrAppend(&encoded_, value);
  return *this;
}

SignatureBuilder& SignatureBuilder::Span(absl::string_view contents, char tag) {
  assert((tag >= 'A' && tag <= 'Z') && "span signature tag must be 'A'..'Z'");
  encoded_.push_back(tag);
  // If the contents starts with a digit or the escape char (!), then escape it.
  absl::StrAppend(&encoded_, contents.size() + 1);
  encoded_.push_back('!');
  encoded_.append(contents.begin(), contents.end());
  return *this;
}

SignatureParser::Type SignatureParser::Next() {
  next_type_ = Type::kError;
  next_tag_ = 0;
  next_ival_ = 0;
  next_sval_ = absl::string_view();
  if (cursor_ == encoded_.end()) {
    next_type_ = Type::kEnd;
    return next_type_;
  }

  next_tag_ = *cursor_;
  absl::string_view::const_iterator ival_begin = cursor_ + 1;
  absl::string_view::const_iterator ival_end = ival_begin;
  while (ival_end != encoded_.end() &&
         ((*ival_end >= '0' && *ival_end <= '9') ||
          (*ival_end == '-' && ival_end == ival_begin))) {
    ++ival_end;
  }

  // No numeric value.
  if (ival_end == ival_begin) {
    return next_type_;
  }

  // Parse ival.
  if (!absl::SimpleAtoi(
          absl::string_view(&(*ival_begin), ival_end - ival_begin),
          &next_ival_)) {
    // Should not be possible.
    return next_type_;
  }

  // For integer components ('_', 'a'..'z'), that is all.
  if (next_tag_ == '_' || (next_tag_ >= 'a' && next_tag_ <= 'z')) {
    next_type_ = Type::kInteger;
    cursor_ = ival_end;
    return next_type_;
  }

  // For string components ('A'..'Z'), extract the string.
  if (next_tag_ >= 'A' && next_tag_ <= 'Z') {
    if (next_ival_ < 0) return next_type_;  // Negative size error.
    absl::string_view::const_iterator sval_begin = ival_end;
    absl::string_view::const_iterator sval_end = sval_begin + next_ival_;
    if (sval_end > encoded_.end()) return next_type_;  // Underrun.

    // Remove escape char if escaped.
    if (next_ival_ == 0 || *sval_begin != '!') {
      next_type_ = Type::kError;
      return next_type_;
    }
    next_ival_ -= 1;
    ++sval_begin;

    next_sval_ = absl::string_view(&(*sval_begin), sval_end - sval_begin);
    cursor_ = sval_end;
    next_type_ = Type::kSpan;
    return next_type_;
  }

  // Otherwise, error.
  return next_type_;
}

bool SignatureParser::SeekTag(char tag) {
  while (next_tag_ != tag && next_type_ != Type::kEnd) {
    Next();
  }
  return next_type_ != Type::kEnd;
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

void RawSignatureMangler::AddUnrecognized() {
  builder_.Span(absl::string_view(), 'U');
}

void RawSignatureMangler::AddAnyReference() {
  // A more constrained ref object would have a non empty span.
  builder_.Span(absl::string_view(), 'O');
}

void RawSignatureMangler::AddShapedNDBuffer(
    AbiConstants::ScalarType element_type, absl::Span<const int> shape) {
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

// -----------------------------------------------------------------------------
// RawSignatureParser
// -----------------------------------------------------------------------------

void RawSignatureParser::Description::ToString(std::string& s) const {
  switch (type) {
    case Type::kBuffer: {
      const char* scalar_type_name = "!BADTYPE!";
      unsigned scalar_type_u = static_cast<unsigned>(buffer.scalar_type);
      if (scalar_type_u >= 0 &&
          scalar_type_u <= AbiConstants::kScalarTypeNames.size()) {
        scalar_type_name = AbiConstants::kScalarTypeNames[static_cast<unsigned>(
            scalar_type_u)];
      }
      absl::StrAppend(&s, "Buffer<", scalar_type_name, "[");
      for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) s.push_back('x');
        if (dims[i] >= 0) {
          absl::StrAppend(&s, dims[i]);
        } else {
          s.push_back('?');
        }
      }
      absl::StrAppend(&s, "]>");
      break;
    }
    case Type::kRefObject:
      absl::StrAppend(&s, "RefObject<?>");
      break;
    default:
      absl::StrAppend(&s, "!UNKNOWN!");
  }
}

absl::optional<std::string> RawSignatureParser::FunctionSignatureToString(
    absl::string_view signature) {
  std::string s;

  bool print_sep = false;
  auto visitor = [&print_sep, &s](const Description& d) {
    if (print_sep) {
      s.append(", ");
    }
    d.ToString(s);
    print_sep = true;
  };
  s.push_back('(');
  VisitInputs(signature, visitor);
  s.append(") -> (");
  print_sep = false;
  VisitResults(signature, visitor);
  s.push_back(')');

  if (!GetError()) {
    return s;
  } else {
    return absl::nullopt;
  }
}

// -----------------------------------------------------------------------------
// SipSignatureMangler
// -----------------------------------------------------------------------------

SipSignatureMangler::SipSignatureMangler() = default;

bool SipSignatureMangler::SetRawSignatureIndex(int raw_signature_index,
                                               absl::Span<const Key> path) {
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
      auto child = absl::make_unique<Value>();
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

absl::optional<SignatureBuilder> SipSignatureMangler::ToFunctionSignature(
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

// -----------------------------------------------------------------------------
// SipSignatureParser
// -----------------------------------------------------------------------------

void SipSignatureParser::ToStringVisitor::IntegerKey(SipSignatureParser& p,
                                                     int k) {
  absl::StrAppend(&s_, indent_, k);
}

void SipSignatureParser::ToStringVisitor::StringKey(SipSignatureParser& p,
                                                    absl::string_view k) {
  absl::StrAppend(&s_, indent_, k);
}

void SipSignatureParser::ToStringVisitor::OpenStruct(SipSignatureParser& p,
                                                     StructType struct_type) {
  absl::StrAppend(&indent_, "  ");
  switch (struct_type) {
    case StructType::kDict:
      close_char_.push_back('}');
      absl::StrAppend(&s_, ":{");
      break;
    case StructType::kSequence:
      close_char_.push_back(']');
      absl::StrAppend(&s_, ":[");
      break;
    default:
      close_char_.push_back('?');
      absl::StrAppend(&s_, ":?");
  }
  absl::StrAppend(&s_, "\n");
}

void SipSignatureParser::ToStringVisitor::CloseStruct(SipSignatureParser& p) {
  if (indent_.size() >= 2) {
    indent_.resize(indent_.size() - 2);
  }
  absl::StrAppend(&s_, indent_);
  s_.push_back(close_char_.back());
  close_char_.pop_back();
  absl::StrAppend(&s_, ",\n");
}

void SipSignatureParser::ToStringVisitor::MapToRawSignatureIndex(
    SipSignatureParser& p, int index) {
  absl::StrAppend(&s_, "=raw(", index, "),\n");
}

}  // namespace iree
