// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/huggingface/postprocessor_json.h"

#include "iree/base/internal/json.h"

//===----------------------------------------------------------------------===//
// TemplateProcessing Parser
//===----------------------------------------------------------------------===//

// Looks up a special token ID from the special_tokens map.
// Returns -1 if not found.
static int32_t iree_tokenizer_lookup_special_token_id(
    iree_string_view_t special_tokens_json, iree_string_view_t token_name) {
  // special_tokens format:
  // {
  //   "[CLS]": {"id": "[CLS]", "ids": [101], "tokens": ["[CLS]"]},
  //   "[SEP]": {"id": "[SEP]", "ids": [102], "tokens": ["[SEP]"]}
  // }
  iree_string_view_t token_entry = iree_string_view_empty();
  iree_status_t status = iree_json_try_lookup_object_value(
      special_tokens_json, token_name, &token_entry);
  if (!iree_status_is_ok(status) || token_entry.size == 0) {
    iree_status_ignore(status);
    return -1;
  }

  // Look up the "ids" array and get the first element.
  iree_string_view_t ids_array = iree_string_view_empty();
  status = iree_json_try_lookup_object_value(token_entry, IREE_SV("ids"),
                                             &ids_array);
  if (!iree_status_is_ok(status) || ids_array.size == 0) {
    iree_status_ignore(status);
    return -1;
  }

  // Get the first element of the ids array.
  iree_string_view_t first_id = iree_string_view_empty();
  status = iree_json_array_get(ids_array, 0, &first_id);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return -1;
  }

  int64_t id_value = 0;
  status = iree_json_parse_int64(first_id, &id_value);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return -1;
  }

  return (int32_t)id_value;
}

// Parses a single template piece.
// Piece format:
//   {"SpecialToken": {"id": "[CLS]", "type_id": 0}}
//   {"Sequence": {"id": "A", "type_id": 0}}
static iree_status_t iree_tokenizer_parse_template_piece(
    iree_string_view_t piece_json, iree_string_view_t special_tokens_json,
    iree_tokenizer_template_piece_t* out_piece) {
  // Validate allowed keys for piece object.
  static const iree_string_view_t kPieceAllowedKeys[] = {
      IREE_SVL("SpecialToken"),
      IREE_SVL("Sequence"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      piece_json, kPieceAllowedKeys, IREE_ARRAYSIZE(kPieceAllowedKeys)));

  memset(out_piece, 0, sizeof(*out_piece));
  out_piece->token_id = -1;

  // Check if it's a SpecialToken.
  iree_string_view_t special_token_obj = iree_string_view_empty();
  iree_status_t status = iree_json_try_lookup_object_value(
      piece_json, IREE_SV("SpecialToken"), &special_token_obj);
  if (iree_status_is_ok(status) && special_token_obj.size > 0) {
    // Validate SpecialToken object keys.
    static const iree_string_view_t kSpecialTokenKeys[] = {
        IREE_SVL("id"),
        IREE_SVL("type_id"),
    };
    IREE_RETURN_IF_ERROR(
        iree_json_validate_object_keys(special_token_obj, kSpecialTokenKeys,
                                       IREE_ARRAYSIZE(kSpecialTokenKeys)));

    out_piece->type = IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL;

    // Get the token name from "id" field.
    iree_string_view_t id_value = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
        special_token_obj, IREE_SV("id"), &id_value));

    // Look up the actual token ID from special_tokens map.
    out_piece->token_id =
        iree_tokenizer_lookup_special_token_id(special_tokens_json, id_value);
    if (out_piece->token_id < 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "special token '%.*s' not found in special_tokens",
          (int)id_value.size, id_value.data);
    }

    // Get type_id (optional, default 0).
    int64_t type_id = 0;
    IREE_RETURN_IF_ERROR(iree_json_try_lookup_int64(
        special_token_obj, IREE_SV("type_id"), 0, &type_id));
    out_piece->type_id = (uint8_t)type_id;

    return iree_ok_status();
  }
  iree_status_ignore(status);

  // Check if it's a Sequence placeholder.
  iree_string_view_t sequence_obj = iree_string_view_empty();
  status = iree_json_try_lookup_object_value(piece_json, IREE_SV("Sequence"),
                                             &sequence_obj);
  if (iree_status_is_ok(status) && sequence_obj.size > 0) {
    // Validate Sequence object keys.
    static const iree_string_view_t kSequenceKeys[] = {
        IREE_SVL("id"),
        IREE_SVL("type_id"),
    };
    IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
        sequence_obj, kSequenceKeys, IREE_ARRAYSIZE(kSequenceKeys)));

    // Get the sequence ID ("A" or "B").
    iree_string_view_t id_value = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(
        iree_json_lookup_object_value(sequence_obj, IREE_SV("id"), &id_value));

    if (iree_string_view_equal(id_value, IREE_SV("A"))) {
      out_piece->type = IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_A;
    } else if (iree_string_view_equal(id_value, IREE_SV("B"))) {
      out_piece->type = IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_B;
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown sequence id '%.*s'", (int)id_value.size,
                              id_value.data);
    }

    // Get type_id (optional, default 0).
    int64_t type_id = 0;
    IREE_RETURN_IF_ERROR(iree_json_try_lookup_int64(
        sequence_obj, IREE_SV("type_id"), 0, &type_id));
    out_piece->type_id = (uint8_t)type_id;

    return iree_ok_status();
  }
  iree_status_ignore(status);

  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "template piece must be SpecialToken or Sequence");
}

// Parses a TemplateProcessing post-processor.
static iree_status_t iree_tokenizer_parse_template_pp(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_postprocessor_t* out_pp) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Validate allowed keys.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("special_tokens"),
      IREE_SVL("single"),
      IREE_SVL("pair"),
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_validate_object_keys(json, kAllowedKeys,
                                         IREE_ARRAYSIZE(kAllowedKeys)));

  // Get special_tokens map (required for looking up token IDs).
  iree_string_view_t special_tokens = iree_string_view_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_try_lookup_object_value(json, IREE_SV("special_tokens"),
                                            &special_tokens));

  // Get single and pair template arrays.
  iree_string_view_t single_array = iree_string_view_empty();
  iree_string_view_t pair_array = iree_string_view_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_try_lookup_object_value(json, IREE_SV("single"),
                                            &single_array));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_json_try_lookup_object_value(json, IREE_SV("pair"), &pair_array));

  // Count pieces.
  iree_host_size_t single_count = 0;
  iree_host_size_t pair_count = 0;
  if (single_array.size > 0 &&
      !iree_string_view_equal(single_array, IREE_SV("null"))) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_json_array_length(single_array, &single_count));
  }
  if (pair_array.size > 0 &&
      !iree_string_view_equal(pair_array, IREE_SV("null"))) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_json_array_length(pair_array, &pair_count));
  }

  if (single_count == 0 && pair_count == 0) {
    // Empty template - treat as NONE.
    iree_tokenizer_postprocessor_initialize_none(out_pp);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Check for overflow in template array size.
  if (single_count > IREE_HOST_SIZE_MAX - pair_count) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "template piece count overflow");
  }
  iree_host_size_t total_count = single_count + pair_count;
  if (total_count >
      IREE_HOST_SIZE_MAX / sizeof(iree_tokenizer_template_piece_t)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "template piece count overflow");
  }
  iree_host_size_t templates_size =
      total_count * sizeof(iree_tokenizer_template_piece_t);

  // Allocate templates array.
  iree_tokenizer_template_piece_t* templates = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, templates_size, (void**)&templates));

  // Parse single template pieces.
  for (iree_host_size_t i = 0; i < single_count; ++i) {
    iree_string_view_t piece_json = iree_string_view_empty();
    iree_status_t status = iree_json_array_get(single_array, i, &piece_json);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(allocator, templates);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
    status = iree_tokenizer_parse_template_piece(piece_json, special_tokens,
                                                 &templates[i]);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(allocator, templates);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
  }

  // Parse pair template pieces.
  for (iree_host_size_t i = 0; i < pair_count; ++i) {
    iree_string_view_t piece_json = iree_string_view_empty();
    iree_status_t status = iree_json_array_get(pair_array, i, &piece_json);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(allocator, templates);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
    status = iree_tokenizer_parse_template_piece(piece_json, special_tokens,
                                                 &templates[single_count + i]);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(allocator, templates);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
  }

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tokenizer_postprocessor_initialize_template(
              templates, single_count, pair_count, allocator, out_pp));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// RobertaProcessing Parser
//===----------------------------------------------------------------------===//

// Parses a RobertaProcessing post-processor.
// Format:
// {
//   "type": "RobertaProcessing",
//   "sep": ["</s>", 2],
//   "cls": ["<s>", 0],
//   "trim_offsets": true,
//   "add_prefix_space": true
// }
static iree_status_t iree_tokenizer_parse_roberta_pp(
    iree_string_view_t json, iree_tokenizer_postprocessor_t* out_pp) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Validate allowed keys.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("cls"),
      IREE_SVL("sep"),
      IREE_SVL("trim_offsets"),
      IREE_SVL("add_prefix_space"),
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_validate_object_keys(json, kAllowedKeys,
                                         IREE_ARRAYSIZE(kAllowedKeys)));

  // Parse cls: ["<s>", 0] - second element is the token ID.
  iree_string_view_t cls_array = iree_string_view_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_lookup_object_value(json, IREE_SV("cls"), &cls_array));

  iree_string_view_t cls_id_str = iree_string_view_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_array_get(cls_array, 1, &cls_id_str));
  int64_t cls_id = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_json_parse_int64(cls_id_str, &cls_id));

  // Parse sep: ["</s>", 2] - second element is the token ID.
  iree_string_view_t sep_array = iree_string_view_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_lookup_object_value(json, IREE_SV("sep"), &sep_array));

  iree_string_view_t sep_id_str = iree_string_view_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_array_get(sep_array, 1, &sep_id_str));
  int64_t sep_id = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_json_parse_int64(sep_id_str, &sep_id));

  // Parse flags.
  iree_tokenizer_roberta_flags_t flags = IREE_TOKENIZER_ROBERTA_FLAG_DEFAULT;

  bool add_prefix_space = true;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_try_lookup_bool(json, IREE_SV("add_prefix_space"), true,
                                    &add_prefix_space));
  if (add_prefix_space) {
    flags |= IREE_TOKENIZER_ROBERTA_FLAG_ADD_PREFIX_SPACE;
  }

  bool trim_offsets = true;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_try_lookup_bool(json, IREE_SV("trim_offsets"), true,
                                    &trim_offsets));
  if (trim_offsets) {
    flags |= IREE_TOKENIZER_ROBERTA_FLAG_TRIM_OFFSETS;
  }

  iree_tokenizer_postprocessor_initialize_roberta(
      (int32_t)cls_id, (int32_t)sep_id, flags, out_pp);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// ByteLevel Parser
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_parse_byte_level_pp(
    iree_string_view_t json, iree_tokenizer_postprocessor_t* out_pp) {
  // Validate allowed keys.
  // add_prefix_space and trim_offsets: accepted but unused. The ByteLevel
  // postprocessor is a no-op for encoding (just passes text through). These
  // keys only affect offset metadata which we don't track.
  // use_regex: encoding-only parameter, fully supported in pre-tokenizer.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("add_prefix_space"),
      IREE_SVL("trim_offsets"),
      IREE_SVL("use_regex"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  iree_tokenizer_postprocessor_initialize_byte_level(0, out_pp);
  return iree_ok_status();
}

// Parses a Sequence post-processor (chain of processors).
// Format:
// {
//   "type": "Sequence",
//   "processors": [
//     {"type": "ByteLevel", ...},
//     {"type": "TemplateProcessing", ...}
//   ]
// }
static iree_status_t iree_tokenizer_parse_sequence_pp(
    iree_string_view_t json, iree_allocator_t allocator,
    iree_tokenizer_postprocessor_t* out_pp) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Validate allowed keys.
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("processors"),
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_validate_object_keys(json, kAllowedKeys,
                                         IREE_ARRAYSIZE(kAllowedKeys)));

  // Get processors array.
  iree_string_view_t processors_array = iree_string_view_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_lookup_object_value(json, IREE_SV("processors"),
                                        &processors_array));

  // Count processors.
  iree_host_size_t count = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_array_length(processors_array, &count));

  if (count == 0) {
    iree_tokenizer_postprocessor_initialize_none(out_pp);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Check for overflow in children array size.
  if (count > IREE_HOST_SIZE_MAX / sizeof(iree_tokenizer_postprocessor_t)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "sequence child count overflow");
  }
  iree_host_size_t children_size =
      count * sizeof(iree_tokenizer_postprocessor_t);

  // Allocate children array.
  iree_tokenizer_postprocessor_t* children = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, children_size, (void**)&children));
  memset(children, 0, children_size);

  // Set up structure immediately so deinitialize handles cleanup on any error.
  out_pp->type = IREE_TOKENIZER_POSTPROCESSOR_SEQUENCE;
  out_pp->config.sequence.count = count;
  out_pp->config.sequence.children = children;
  out_pp->config.sequence.allocator = allocator;

  // Parse each child processor.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    iree_string_view_t child_json = iree_string_view_empty();
    status = iree_json_array_get(processors_array, i, &child_json);
    if (iree_status_is_ok(status)) {
      status = iree_tokenizer_postprocessor_parse_field(child_json, allocator,
                                                        &children[i]);
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_tokenizer_postprocessor_deinitialize(out_pp);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_postprocessor_parse_field(
    iree_string_view_t pp_json, iree_allocator_t allocator,
    iree_tokenizer_postprocessor_t* out_pp) {
  IREE_ASSERT_ARGUMENT(out_pp);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_postprocessor_initialize_none(out_pp);

  // Look up the type field.
  iree_string_view_t type_value = iree_string_view_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_lookup_object_value(pp_json, IREE_SV("type"), &type_value));

  // Dispatch based on type.
  iree_status_t status = iree_ok_status();
  if (iree_string_view_equal(type_value, IREE_SV("TemplateProcessing"))) {
    status = iree_tokenizer_parse_template_pp(pp_json, allocator, out_pp);
  } else if (iree_string_view_equal(type_value, IREE_SV("RobertaProcessing"))) {
    status = iree_tokenizer_parse_roberta_pp(pp_json, out_pp);
  } else if (iree_string_view_equal(type_value, IREE_SV("ByteLevel"))) {
    status = iree_tokenizer_parse_byte_level_pp(pp_json, out_pp);
  } else if (iree_string_view_equal(type_value, IREE_SV("Sequence"))) {
    status = iree_tokenizer_parse_sequence_pp(pp_json, allocator, out_pp);
  } else if (iree_string_view_equal(type_value, IREE_SV("BertProcessing"))) {
    // BertProcessing is similar to TemplateProcessing but with fixed format.
    // For now, treat as unimplemented - real BERT models use
    // TemplateProcessing.
    status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "BertProcessing not yet supported, use "
                              "TemplateProcessing equivalent");
  } else {
    status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported post_processor type '%.*s'",
                              (int)type_value.size, type_value.data);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_tokenizer_postprocessor_parse_json(
    iree_string_view_t json_root, iree_allocator_t allocator,
    iree_tokenizer_postprocessor_t* out_pp) {
  IREE_ASSERT_ARGUMENT(out_pp);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_postprocessor_initialize_none(out_pp);

  // Look up the post_processor field (optional).
  iree_string_view_t pp_value = iree_string_view_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_try_lookup_object_value(
              json_root, IREE_SV("post_processor"), &pp_value));

  if (pp_value.size == 0 || iree_string_view_equal(pp_value, IREE_SV("null"))) {
    // null or missing - return NONE (no special tokens).
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  iree_status_t status =
      iree_tokenizer_postprocessor_parse_field(pp_value, allocator, out_pp);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
