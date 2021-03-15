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

#ifndef IREE_COMPILER_BINDINGS_SIP_UTILS_SIGNATURE_BUILDER_H_
#define IREE_COMPILER_BINDINGS_SIP_UTILS_SIGNATURE_BUILDER_H_

#include <array>
#include <cassert>
#include <map>
#include <string>

#include "iree/compiler/Bindings/SIP/Utils/Signature.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace SIP {

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
  SignatureBuilder& Span(StringRef contents, char tag);

  // Appends to another builder as a sub-span with the given tag.
  const SignatureBuilder& AppendTo(SignatureBuilder& other, char tag) const {
    other.Span(encoded_, tag);
    return *this;
  }

 private:
  std::string encoded_;
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
                         ArrayRef<int> shape);

  void AddScalar(AbiConstants::ScalarType type);

  const SignatureBuilder& builder() const { return builder_; }

 private:
  SignatureBuilder builder_;
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
    Key(StringRef skey) : skey_(skey), ikey_(-1) {}
    Key(const char* skey) : skey_(skey), ikey_(-1) {}

    bool is_integer_key() const { return ikey_ >= 0; }
    bool is_string_key() const { return ikey_ < 0; }

    IndexMode index_mode() const {
      return is_integer_key() ? IndexMode::kSequence : IndexMode::kDict;
    }

    int ikey() const { return ikey_; }
    StringRef skey() const { return skey_; }

    bool operator==(const Key& other) const {
      return ikey_ == other.ikey_ && skey_ == other.skey_;
    }
    bool operator<(const Key& other) const {
      return (ikey_ != other.ikey_) ? (ikey_ < other.ikey_)
                                    : (skey_ < other.skey_);
    }

   private:
    StringRef skey_;
    int ikey_;
  };
  SipSignatureMangler();

  // Sets the raw signature index at a structure leaf as identified by path.
  // Returns whether the path and index are valid.
  bool SetRawSignatureIndex(int raw_signature_index, ArrayRef<Key> path);

  // Emits a signature for the resulting structure, which will typically
  // be embedded in a full function signature as either inputs or results.
  llvm::Optional<SignatureBuilder> ToStructureSignature() const {
    SignatureBuilder sb;
    if (!ToStructureSignature(&sb, &root_)) {
      return llvm::None;
    }
    return sb;
  }

  // Generates a full function signature from structured inputs and results.
  static llvm::Optional<SignatureBuilder> ToFunctionSignature(
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

}  // namespace SIP
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_BINDINGS_SIP_UTILS_SIGNATURE_BUILDER_H_
