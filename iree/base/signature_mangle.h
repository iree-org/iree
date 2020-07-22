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

#ifndef IREE_BASE_SIGNATURE_MANGLE_H_
#define IREE_BASE_SIGNATURE_MANGLE_H_

#include <array>
#include <cassert>
#include <map>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "iree/base/status.h"

// Name mangling/demangling for function and type signatures.
namespace iree {

namespace AbiConstants {

// Canonical integer mappings are maintained for core scalar type codes
// since they change infrequently and are used everywhere.
// Generally, favor adding a custom type vs extending this arbitrarily.
enum class ScalarType : unsigned {
  kIeeeFloat32 = 0,
  kIeeeFloat16 = 1,
  kIeeeFloat64 = 2,
  kGoogleBfloat16 = 3,
  kSint8 = 4,
  kSint16 = 5,
  kSint32 = 6,
  kSint64 = 7,
  kUint8 = 8,
  kUint16 = 9,
  kUint32 = 10,
  kUint64 = 11,
  kMaxScalarType = 11,
};

// Array that maps ScalarType codes to the size in bytes.
extern const std::array<size_t,
                        static_cast<unsigned>(ScalarType::kMaxScalarType) + 1>
    kScalarTypeSize;

extern const std::array<const char*,
                        static_cast<unsigned>(ScalarType::kMaxScalarType) + 1>
    kScalarTypeNames;

}  // namespace AbiConstants

// Builds up a signature string from components.
// The signature syntax is a sequence of Integer or Span fields:
//   integer_tag ::= '_' | [a-z]
//   integer ::= integer_tag ('-')?[0-9]+
//   span_tag ::= [A-Z]
//   span ::= span_tag (LENGTH:[0-9]+) .{LENGTH}
//
//   component ::= integer-component | span-component
//   integer-component ::= integer-tag integer
//   span-component ::= span-tag length '!' contents
//     # (Where 'length' encoded the length in bytes of 'contents' plus 1 for
//     # the '!'.
//
// Low-level lexical primitives:
//   integer ::= -?[0-9]+
//   length ::= [0-9]+
//   integer-tag ::= '_' | [a-z]
//   span-tag ::= [A-Z]
class SignatureBuilder {
 public:
  SignatureBuilder() = default;
  ~SignatureBuilder() = default;

  std::string& encoded() { return encoded_; }
  const std::string& encoded() const { return encoded_; }

  // Appends an integer component with the given tag (or generic integer
  // tag '_'). The tag must be a lower-case ascii letter between 'a'..'z'
  // inclusive.
  SignatureBuilder& Integer(int value, char tag = '_');

  // Appends a literal span with a tag.
  // The tag must be an upper-case ascii letter between 'A'..'Z' inclusive.
  SignatureBuilder& Span(absl::string_view contents, char tag);

  // Appends to another builder as a sub-span with the given tag.
  const SignatureBuilder& AppendTo(SignatureBuilder& other, char tag) const {
    other.Span(encoded_, tag);
    return *this;
  }

 private:
  std::string encoded_;
};

// Parses a signature produced by SignatureBuilder.
// The parser works field-by-field and it is up to the caller to handle nesting
// by handling nested SignatureParsers (typically by calling nested()).
class SignatureParser {
 public:
  enum class Type {
    kEnd,
    kInteger,
    kSpan,
    kError,
  };

  explicit SignatureParser(absl::string_view encoded)
      : encoded_(encoded), cursor_(encoded_.begin()) {
    Next();
  }

  // Gets the next component from the signature.
  Type Next();

  // Seek to the next field with the given tag (potentially this one).
  // Returns true if found. If false, the parser will be at kEnd.
  bool SeekTag(char tag);

  bool end_or_error() const {
    return next_type_ == Type::kEnd || next_type_ == Type::kError;
  }
  Type type() const { return next_type_; }
  char tag() const { return next_tag_; }
  int ival() const { return next_ival_; }
  absl::string_view sval() const { return next_sval_; }
  SignatureParser nested() const { return SignatureParser(next_sval_); }

 private:
  absl::string_view encoded_;

  // Cursor is always positioned at the start of the next component.
  absl::string_view::const_iterator cursor_;

  Type next_type_;
  int next_ival_;
  absl::string_view next_sval_;
  char next_tag_;
};

// -----------------------------------------------------------------------------
// Raw signatures
// -----------------------------------------------------------------------------

// Mangles raw function signatures.
// See docs/design_docs/function_abi.md.
class RawSignatureMangler {
 public:
  static SignatureBuilder ToFunctionSignature(const SignatureBuilder& inputs,
                                              const SignatureBuilder& results);

  // Combines mangled input and result signatures into a function signature.
  static SignatureBuilder ToFunctionSignature(
      const RawSignatureMangler& inputs, const RawSignatureMangler& results) {
    return ToFunctionSignature(inputs.builder(), results.builder());
  }

  // Adds an unrecognized type. By default, this is an empty span, but in the
  // future, it may contain some further description.
  void AddUnrecognized();

  // Adds an unconstrained reference-type object.
  void AddAnyReference();

  // Adds a shaped nd buffer operand with the given element type and shape.
  // Unknown dims should be -1.
  // This is the common case for external interfacing and requires a fully
  // ranked shape.
  void AddShapedNDBuffer(AbiConstants::ScalarType element_type,
                         absl::Span<const int> shape);

  void AddScalar(AbiConstants::ScalarType type);

  const SignatureBuilder& builder() const { return builder_; }

 private:
  SignatureBuilder builder_;
};

// Parses function signatures generated by RawSignatureMangler.
class RawSignatureParser {
 public:
  using DimVector = absl::InlinedVector<int, 4>;

  enum class Type {
    kBuffer = 0,
    kRefObject = 1,
    kScalar = 2,
  };

  // Description of an input or result.
  struct Description {
    // Type category of the argument.
    Type type;

    // For shaped types, this is the corresponding dims.
    DimVector dims;

    union {
      // Further details for Type == kBuffer.
      struct {
        AbiConstants::ScalarType scalar_type;
      } buffer;
      // Further details for Type == kScalar.
      struct {
        AbiConstants::ScalarType type;
      } scalar;
    };

    // Human readable description.
    void ToString(std::string& s) const;
  };

  using Visitor = std::function<void(const Description&)>;

  void VisitInputs(absl::string_view signature, Visitor visitor) {
    SignatureParser sp(signature);
    if (!sp.SeekTag('I')) {
      SetError("Inputs span not found");
      return;
    }
    auto nested = sp.nested();
    return Visit(visitor, nested);
  }

  void VisitResults(absl::string_view signature, Visitor visitor) {
    SignatureParser sp(signature);
    if (!sp.SeekTag('R')) {
      SetError("Results span not found");
      return;
    }
    auto nested = sp.nested();
    return Visit(visitor, nested);
  }

  // Produces a human readable function signature from the encoded form.
  // Does not return a value on error.
  absl::optional<std::string> FunctionSignatureToString(
      absl::string_view signature);

  // If the parser is in an error state, accesses the error.
  const absl::optional<std::string>& GetError() { return error_; }
  void SetError(std::string error) {
    if (!error_) error_ = std::move(error);
  }

 private:
  void Visit(Visitor& v, SignatureParser& item_parser) {
    Description d;
    while (!item_parser.end_or_error() && !error_) {
      // Reset shared fields.
      d.dims.clear();

      switch (item_parser.tag()) {
        case 'B':
          if (!FillBuffer(d, SignatureParser(item_parser.nested()))) {
            return;
          }
          break;
        case 'O':
          if (!FillRefObject(d, SignatureParser(item_parser.nested()))) {
            return;
          }
          break;
        case 'S':
          if (!FillScalar(d, SignatureParser(item_parser.nested()))) {
            return;
          }
          break;
        default:
          SetError("Unrecognized raw tag");
          return;
      }

      v(d);
      item_parser.Next();
    }
  }

  bool FillScalar(Description& d, SignatureParser p) {
    d.type = Type::kScalar;
    d.buffer.scalar_type = AbiConstants::ScalarType::kIeeeFloat32;  // Default
    while (!p.end_or_error()) {
      switch (p.tag()) {
        case 't':
          if (p.ival() < 0 ||
              p.ival() >
                  static_cast<int>(AbiConstants::ScalarType::kMaxScalarType)) {
            SetError("Illegal ScalarType code");
            return false;
          }
          d.buffer.scalar_type =
              static_cast<AbiConstants::ScalarType>(p.ival());
          break;
        default:
          SetError("Unrecognized scalar field tag");
          return false;
      }
      p.Next();
    }
    return true;
  }

  bool FillBuffer(Description& d, SignatureParser p) {
    d.type = Type::kBuffer;
    d.buffer.scalar_type = AbiConstants::ScalarType::kIeeeFloat32;  // Default
    while (!p.end_or_error()) {
      switch (p.tag()) {
        case 't':
          if (p.ival() < 0 ||
              p.ival() >
                  static_cast<int>(AbiConstants::ScalarType::kMaxScalarType)) {
            SetError("Illegal ScalarType code");
            return false;
          }
          d.buffer.scalar_type =
              static_cast<AbiConstants::ScalarType>(p.ival());
          break;
        case 'd':
          d.dims.push_back(p.ival());
          break;
        default:
          SetError("Unrecognized buffer field tag");
          return false;
      }
      p.Next();
    }
    return true;
  }

  bool FillRefObject(Description& d, SignatureParser p) {
    d.type = Type::kRefObject;
    while (!p.end_or_error()) {
      switch (p.tag()) {
        default:
          SetError("Unrecognized ref object field tag");
          return false;
      }
      p.Next();
    }
    return true;
  }

  absl::optional<std::string> error_;
};

// -----------------------------------------------------------------------------
// Sip signatures
// -----------------------------------------------------------------------------

// Mangles function signatures according to the Sip (Structured Index Path) V1
// scheme.
//
// Mangler for the 'sip' ABI. See docs/design_docs/function_abi.md in the
// documentation.
class SipSignatureMangler {
 public:
  enum class IndexMode {
    kNone,
    kSequence,
    kDict,
  };

  class Key {
   public:
    Key(int ikey) : skey_(), ikey_(ikey) { assert(ikey_ >= 0); }
    Key(absl::string_view skey) : skey_(skey), ikey_(-1) {}
    Key(const char* skey) : skey_(skey), ikey_(-1) {}

    bool is_integer_key() const { return ikey_ >= 0; }
    bool is_string_key() const { return ikey_ < 0; }

    IndexMode index_mode() const {
      return is_integer_key() ? IndexMode::kSequence : IndexMode::kDict;
    }

    int ikey() const { return ikey_; }
    absl::string_view skey() const { return skey_; }

    bool operator==(const Key& other) const {
      return ikey_ == other.ikey_ && skey_ == other.skey_;
    }
    bool operator<(const Key& other) const {
      return (ikey_ != other.ikey_) ? (ikey_ < other.ikey_)
                                    : (skey_ < other.skey_);
    }

   private:
    absl::string_view skey_;
    int ikey_;
  };
  SipSignatureMangler();

  // Sets the raw signature index at a structure leaf as identified by path.
  // Returns whether the path and index are valid.
  bool SetRawSignatureIndex(int raw_signature_index,
                            absl::Span<const Key> path);

  // Emits a signature for the resulting structure, which will typically
  // be embedded in a full function signature as either inputs or results.
  absl::optional<SignatureBuilder> ToStructureSignature() const {
    SignatureBuilder sb;
    if (!ToStructureSignature(&sb, &root_)) {
      return absl::optional<SignatureBuilder>();
    }
    return sb;
  }

  // Generates a full function signature from structured inputs and results.
  static absl::optional<SignatureBuilder> ToFunctionSignature(
      const SipSignatureMangler& inputs_struct,
      const SipSignatureMangler& results_struct);

 private:
  struct Value {
    // If this is a leaf, then this will be >= 0 and maps to the flat input/
    // result index in the raw signature.
    int raw_signature_index = -1;

    // Whether the value is being indexed as a sequence or a dict.
    IndexMode index_mode = IndexMode::kNone;

    // If not a leaf, then this is the children.
    std::map<Key, std::unique_ptr<Value>> children;
  };

  bool ToStructureSignature(SignatureBuilder* sb, const Value* level) const;
  Value root_;
};

// Parser for signatures generated by SipSignatureMangler.
// This uses a Visitor interface to walk either input or result structs.
//
// Mangler for the 'sip' ABI. See docs/design_docs/function_abi.md in the
// documentation.
class SipSignatureParser {
 public:
  enum class StructType {
    kSequence,
    kDict,
  };

  template <typename Visitor>
  struct VisitorAdapter {
    VisitorAdapter(SipSignatureParser& p, Visitor& v) : p(p), v(v) {}

    void IntegerKey(int k) { v.IntegerKey(p, k); }
    void StringKey(absl::string_view k) { v.StringKey(p, k); }

    void OpenStruct(StructType struct_type) { v.OpenStruct(p, struct_type); }
    void CloseStruct() { v.CloseStruct(p); }

    void MapToRawSignatureIndex(int index) {
      v.MapToRawSignatureIndex(p, index);
    }

    SipSignatureParser& p;
    Visitor& v;
  };

  class ToStringVisitor {
   public:
    void IntegerKey(SipSignatureParser& p, int k);
    void StringKey(SipSignatureParser& p, absl::string_view k);
    void OpenStruct(SipSignatureParser& p, StructType struct_type);
    void CloseStruct(SipSignatureParser& p);
    void MapToRawSignatureIndex(SipSignatureParser& p, int index);

    std::string& s() { return s_; }

   private:
    std::string s_;
    std::string indent_;
    std::string close_char_;
  };

  template <typename Visitor>
  void VisitInputs(Visitor& v, absl::string_view signature) {
    SignatureParser sp(signature);
    if (!sp.SeekTag('I')) {
      return SetError("Inputs struct not found");
    }
    VisitorAdapter<Visitor> va(*this, v);
    auto nested = sp.nested();
    return Visit(va, nested, false);
  }

  template <typename Visitor>
  void VisitResults(Visitor& v, absl::string_view signature) {
    SignatureParser sp(signature);
    if (!sp.SeekTag('R')) {
      return SetError("Results struct not found");
    }
    VisitorAdapter<Visitor> va(*this, v);
    auto nested = sp.nested();
    return Visit(va, nested, false);
  }

  // If the parser is in an error state, accesses the error.
  const absl::optional<std::string>& GetError() { return error_; }
  void SetError(std::string error) {
    if (!error_) error_ = std::move(error);
  }

 private:
  template <typename Visitor>
  void Visit(VisitorAdapter<Visitor>& v, SignatureParser& struct_parser,
             bool allow_key);

  absl::optional<std::string> error_;
};

template <typename Visitor>
void SipSignatureParser::Visit(VisitorAdapter<Visitor>& v,
                               SignatureParser& struct_parser,
                               bool global_allow_key) {
  bool allow_key;
  bool allow_value;

  auto reset_state = [&]() {
    allow_key = global_allow_key;
    allow_value = !allow_key;
  };
  reset_state();

  while (!struct_parser.end_or_error() && !error_) {
    switch (struct_parser.tag()) {
      case 'k':
        if (!allow_key) {
          return SetError("Struct key not allowed here");
        }
        allow_key = false;
        allow_value = true;
        v.IntegerKey(struct_parser.ival());
        break;
      case 'K':
        if (!allow_key) {
          return SetError("Struct key not allowed here");
        }
        allow_key = false;
        allow_value = true;
        v.StringKey(struct_parser.sval());
        break;
      case '_':
        if (!allow_value) {
          return SetError("Value not allowed here");
        }
        v.MapToRawSignatureIndex(struct_parser.ival());
        reset_state();
        break;
      case 'S':
      case 'D': {
        if (!allow_value) {
          return SetError("Value not allowed here");
        }
        v.OpenStruct(struct_parser.tag() == 'S' ? StructType::kSequence
                                                : StructType::kDict);
        SignatureParser child_struct_parser(struct_parser.sval());
        Visit(v, child_struct_parser, true);
        v.CloseStruct();
        reset_state();
        break;
      }
      default:
        return SetError("Unrecognized tag");
    }
    struct_parser.Next();
  }

  if (struct_parser.type() == SignatureParser::Type::kError) {
    return SetError("Syntax error in signature");
  }
}

}  // namespace iree

#endif  // IREE_BASE_SIGNATURE_MANGLE_H_
