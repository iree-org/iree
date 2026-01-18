// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/huggingface/tokenizer_json.h"

#include "iree/base/api.h"
#include "iree/base/internal/json.h"
#include "iree/tokenizer/huggingface/bpe_json.h"
#include "iree/tokenizer/huggingface/unigram_json.h"
#include "iree/tokenizer/huggingface/wordpiece_json.h"

// Tokenizer model type inferred from JSON structure.
typedef enum iree_tokenizer_model_type_e {
  IREE_TOKENIZER_MODEL_TYPE_UNKNOWN = 0,
  IREE_TOKENIZER_MODEL_TYPE_BPE,
  IREE_TOKENIZER_MODEL_TYPE_WORDPIECE,
  IREE_TOKENIZER_MODEL_TYPE_UNIGRAM,
} iree_tokenizer_model_type_t;

// Infers the tokenizer type from model structure when model.type is absent.
// Real HuggingFace tokenizer.json files often omit model.type.
static iree_status_t iree_tokenizer_json_infer_type(
    iree_string_view_t model, iree_tokenizer_model_type_t* out_type) {
  *out_type = IREE_TOKENIZER_MODEL_TYPE_UNKNOWN;

  // BPE tokenizers have a "merges" array.
  iree_string_view_t merges;
  IREE_RETURN_IF_ERROR(
      iree_json_try_lookup_object_value(model, IREE_SV("merges"), &merges));
  if (merges.size > 0) {
    *out_type = IREE_TOKENIZER_MODEL_TYPE_BPE;
    return iree_ok_status();
  }

  // WordPiece tokenizers have "continuing_subword_prefix" (typically "##").
  iree_string_view_t prefix;
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      model, IREE_SV("continuing_subword_prefix"), &prefix));
  if (prefix.size > 0) {
    *out_type = IREE_TOKENIZER_MODEL_TYPE_WORDPIECE;
    return iree_ok_status();
  }

  // Unigram tokenizers have "unk_id" (integer, not string).
  iree_string_view_t unk_id_value;
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      model, IREE_SV("unk_id"), &unk_id_value));
  if (unk_id_value.size > 0) {
    *out_type = IREE_TOKENIZER_MODEL_TYPE_UNIGRAM;
    return iree_ok_status();
  }

  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_tokenizer_from_huggingface_json(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_t** out_tokenizer) {
  IREE_ASSERT_ARGUMENT(out_tokenizer);
  *out_tokenizer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (json.size == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "empty JSON input");
  }

  // Look up the model object.
  iree_string_view_t model;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_lookup_object_value(json, IREE_SV("model"), &model));

  // Determine model type from explicit type string or infer from structure.
  iree_string_view_t type_str = iree_string_view_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_try_lookup_object_value(model, IREE_SV("type"), &type_str));

  iree_tokenizer_model_type_t model_type = IREE_TOKENIZER_MODEL_TYPE_UNKNOWN;
  if (iree_string_view_equal(type_str, IREE_SV("WordPiece"))) {
    model_type = IREE_TOKENIZER_MODEL_TYPE_WORDPIECE;
  } else if (iree_string_view_equal(type_str, IREE_SV("BPE"))) {
    model_type = IREE_TOKENIZER_MODEL_TYPE_BPE;
  } else if (iree_string_view_equal(type_str, IREE_SV("Unigram"))) {
    model_type = IREE_TOKENIZER_MODEL_TYPE_UNIGRAM;
  } else if (type_str.size == 0) {
    // No explicit type - infer from structure.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_tokenizer_json_infer_type(model, &model_type));
  }

  // Dispatch to type-specific factory.
  iree_status_t status = iree_ok_status();
  switch (model_type) {
    case IREE_TOKENIZER_MODEL_TYPE_BPE:
      status = iree_tokenizer_from_bpe_json(json, allocator, out_tokenizer);
      break;
    case IREE_TOKENIZER_MODEL_TYPE_WORDPIECE:
      status =
          iree_tokenizer_from_wordpiece_json(json, allocator, out_tokenizer);
      break;
    case IREE_TOKENIZER_MODEL_TYPE_UNIGRAM:
      status = iree_tokenizer_from_unigram_json(json, allocator, out_tokenizer);
      break;
    case IREE_TOKENIZER_MODEL_TYPE_UNKNOWN:
      // Could not determine type.
      if (type_str.size > 0) {
        status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                  "unsupported tokenizer model type: '%.*s'; "
                                  "supported types are BPE, WordPiece, Unigram",
                                  (int)type_str.size, type_str.data);
      } else {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "could not determine tokenizer type: model.type is missing and "
            "structure does not match BPE (has merges) or WordPiece "
            "(has continuing_subword_prefix)");
      }
      break;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}
