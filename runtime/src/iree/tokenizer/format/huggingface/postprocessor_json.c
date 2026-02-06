// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/format/huggingface/postprocessor_json.h"

#include "iree/base/internal/json.h"

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Parses a [string, uint32] tuple from a JSON array value.
// Used for BertProcessing and RobertaProcessing sep/cls fields.
static iree_status_t iree_tokenizer_parse_token_tuple(
    iree_string_view_t tuple_value, const char* field_name,
    iree_tokenizer_token_id_t* out_token_id) {
  *out_token_id = 0;
  iree_host_size_t array_length = 0;
  IREE_RETURN_IF_ERROR(iree_json_array_length(tuple_value, &array_length));
  if (array_length != 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "post_processor %s must be a [string, id] tuple "
                            "(got array of length %zu)",
                            field_name, array_length);
  }
  // Element 0 is the token string (we don't need it — just the ID).
  // Element 1 is the numeric token ID.
  iree_string_view_t id_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_array_get(tuple_value, 1, &id_value));
  int64_t id = 0;
  IREE_RETURN_IF_ERROR(iree_json_parse_int64(id_value, &id));
  *out_token_id = (iree_tokenizer_token_id_t)id;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// BertProcessing
//===----------------------------------------------------------------------===//

// JSON: {"type":"BertProcessing","sep":["[SEP]",102],"cls":["[CLS]",101]}
//
// Single: [CLS] $A [SEP]
// Pair:   [CLS] $A [SEP] $B [SEP]
// Type IDs: sequence A = 0, sequence B = 1.
// The infix [SEP] gets type_id = 0, the suffix [SEP] gets type_id = 1.
static iree_status_t iree_tokenizer_parse_bert_postprocessor(
    iree_string_view_t json,
    iree_tokenizer_postprocessor_t* out_postprocessor) {
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("sep"),
      IREE_SVL("cls"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  // Parse sep and cls tuples.
  iree_string_view_t sep_value = iree_string_view_empty();
  iree_string_view_t cls_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_object_value(json, IREE_SV("sep"), &sep_value));
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_object_value(json, IREE_SV("cls"), &cls_value));

  iree_tokenizer_token_id_t sep_id = 0;
  iree_tokenizer_token_id_t cls_id = 0;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_parse_token_tuple(sep_value, "sep", &sep_id));
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_parse_token_tuple(cls_value, "cls", &cls_id));

  // Build single template: [CLS] $A [SEP]
  iree_tokenizer_postprocessor_template_t single = {0};
  single.prefix_count = 1;
  single.suffix_count = 1;
  single.sequence_a_type_id = 0;
  single.token_ids[0] = cls_id;  // prefix
  single.token_ids[1] = sep_id;  // suffix
  single.type_ids[0] = 0;        // [CLS] type_id
  single.type_ids[1] = 0;        // [SEP] type_id

  // Build pair template: [CLS] $A [SEP] $B [SEP]
  iree_tokenizer_postprocessor_template_t pair = {0};
  pair.prefix_count = 1;
  pair.infix_count = 1;
  pair.suffix_count = 1;
  pair.sequence_a_type_id = 0;
  pair.sequence_b_type_id = 1;
  pair.token_ids[0] = cls_id;  // prefix [CLS]
  pair.token_ids[1] = sep_id;  // infix [SEP]
  pair.token_ids[2] = sep_id;  // suffix [SEP]
  pair.type_ids[0] = 0;        // [CLS] type_id
  pair.type_ids[1] = 0;        // infix [SEP] type_id (belongs to sequence A)
  pair.type_ids[2] = 1;        // suffix [SEP] type_id (belongs to sequence B)

  return iree_tokenizer_postprocessor_initialize(
      &single, &pair, IREE_TOKENIZER_POSTPROCESSOR_FLAG_NONE,
      out_postprocessor);
}

//===----------------------------------------------------------------------===//
// RobertaProcessing
//===----------------------------------------------------------------------===//

// JSON: {"type":"RobertaProcessing","sep":["</s>",2],"cls":["<s>",0],
//        "trim_offsets":true,"add_prefix_space":true}
//
// Single: <cls> $A <sep>
// Pair:   <cls> $A <sep><sep> $B <sep>
// All type_ids are 0 (RoBERTa does not use segment IDs).
static iree_status_t iree_tokenizer_parse_roberta_postprocessor(
    iree_string_view_t json,
    iree_tokenizer_postprocessor_t* out_postprocessor) {
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("sep"),
      IREE_SVL("cls"),
      IREE_SVL("trim_offsets"),
      IREE_SVL("add_prefix_space"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  // Parse required sep and cls tuples.
  iree_string_view_t sep_value = iree_string_view_empty();
  iree_string_view_t cls_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_object_value(json, IREE_SV("sep"), &sep_value));
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_object_value(json, IREE_SV("cls"), &cls_value));

  iree_tokenizer_token_id_t sep_id = 0;
  iree_tokenizer_token_id_t cls_id = 0;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_parse_token_tuple(sep_value, "sep", &sep_id));
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_parse_token_tuple(cls_value, "cls", &cls_id));

  // Parse required booleans (no serde default in HF).
  bool trim_offsets = false;
  bool add_prefix_space = false;
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_bool(json, IREE_SV("trim_offsets"), &trim_offsets));
  IREE_RETURN_IF_ERROR(iree_json_lookup_bool(json, IREE_SV("add_prefix_space"),
                                             &add_prefix_space));
  iree_tokenizer_postprocessor_flags_t flags =
      IREE_TOKENIZER_POSTPROCESSOR_FLAG_NONE;
  if (trim_offsets) flags |= IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS;
  if (add_prefix_space) {
    flags |= IREE_TOKENIZER_POSTPROCESSOR_FLAG_ADD_PREFIX_SPACE;
  }

  // Build single template: <cls> $A <sep>
  iree_tokenizer_postprocessor_template_t single = {0};
  single.prefix_count = 1;
  single.suffix_count = 1;
  single.sequence_a_type_id = 0;
  single.token_ids[0] = cls_id;
  single.token_ids[1] = sep_id;
  // All type_ids are 0 (default from zero-init).

  // Build pair template: <cls> $A <sep><sep> $B <sep>
  iree_tokenizer_postprocessor_template_t pair = {0};
  pair.prefix_count = 1;
  pair.infix_count = 2;
  pair.suffix_count = 1;
  pair.sequence_a_type_id = 0;
  pair.sequence_b_type_id = 0;
  pair.token_ids[0] = cls_id;  // prefix
  pair.token_ids[1] = sep_id;  // first infix <sep> (ends sequence A)
  pair.token_ids[2] = sep_id;  // second infix <sep> (starts sequence B)
  pair.token_ids[3] = sep_id;  // suffix <sep>
  // All type_ids are 0 (default from zero-init).

  return iree_tokenizer_postprocessor_initialize(&single, &pair, flags,
                                                 out_postprocessor);
}

//===----------------------------------------------------------------------===//
// ByteLevel
//===----------------------------------------------------------------------===//

// JSON: {"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,
//        "use_regex":true}
//
// No special tokens. Only offset trimming behavior.
static iree_status_t iree_tokenizer_parse_byte_level_postprocessor(
    iree_string_view_t json,
    iree_tokenizer_postprocessor_t* out_postprocessor) {
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("add_prefix_space"),
      IREE_SVL("trim_offsets"),
      IREE_SVL("use_regex"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  // Parse required booleans (no serde default in HF).
  bool trim_offsets = false;
  bool add_prefix_space = false;
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_bool(json, IREE_SV("trim_offsets"), &trim_offsets));
  IREE_RETURN_IF_ERROR(iree_json_lookup_bool(json, IREE_SV("add_prefix_space"),
                                             &add_prefix_space));
  // use_regex is informational only for post-processing; not stored.

  iree_tokenizer_postprocessor_flags_t flags =
      IREE_TOKENIZER_POSTPROCESSOR_FLAG_NONE;
  if (trim_offsets) flags |= IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS;
  if (add_prefix_space) {
    flags |= IREE_TOKENIZER_POSTPROCESSOR_FLAG_ADD_PREFIX_SPACE;
  }

  iree_tokenizer_postprocessor_template_t empty = {0};
  return iree_tokenizer_postprocessor_initialize(&empty, /*pair_template=*/NULL,
                                                 flags, out_postprocessor);
}

//===----------------------------------------------------------------------===//
// TemplateProcessing
//===----------------------------------------------------------------------===//

// Phase tracking for walking piece arrays.
typedef enum {
  IREE_TEMPLATE_PHASE_PREFIX = 0,
  IREE_TEMPLATE_PHASE_INFIX,
  IREE_TEMPLATE_PHASE_SUFFIX,
} iree_template_phase_t;

// Context for compiling a single template (single or pair) from its piece
// array.
typedef struct iree_template_compile_context_t {
  iree_tokenizer_postprocessor_template_t* template_out;
  iree_string_view_t special_tokens_value;  // The special_tokens object JSON.
  iree_template_phase_t phase;
  bool seen_sequence_a;
  bool seen_sequence_b;
  iree_status_t status;
} iree_template_compile_context_t;

// Looks up a special token name in the special_tokens map and emits its IDs
// into the template at the appropriate phase position.
static iree_status_t iree_tokenizer_template_emit_special_token(
    iree_template_compile_context_t* context, iree_string_view_t token_name,
    uint32_t type_id) {
  iree_tokenizer_postprocessor_template_t* t = context->template_out;

  // Look up the special token in the map.
  iree_string_view_t token_entry = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      context->special_tokens_value, token_name, &token_entry));

  // Get the "ids" array.
  iree_string_view_t ids_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_object_value(token_entry, IREE_SV("ids"), &ids_value));

  iree_host_size_t ids_count = 0;
  IREE_RETURN_IF_ERROR(iree_json_array_length(ids_value, &ids_count));
  if (ids_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "special token \"%.*s\" has empty ids array",
                            (int)token_name.size, token_name.data);
  }

  // Emit each ID into the current phase.
  for (iree_host_size_t i = 0; i < ids_count; ++i) {
    iree_string_view_t id_element = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_json_array_get(ids_value, i, &id_element));
    int64_t id_value_int = 0;
    IREE_RETURN_IF_ERROR(iree_json_parse_int64(id_element, &id_value_int));

    uint8_t total = iree_tokenizer_postprocessor_template_total_count(t);
    if (total >= IREE_TOKENIZER_POSTPROCESSOR_MAX_PIECES) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "template exceeds maximum of %u special token pieces",
          (unsigned)IREE_TOKENIZER_POSTPROCESSOR_MAX_PIECES);
    }

    // Insert at current position within the phase.
    t->token_ids[total] = (iree_tokenizer_token_id_t)id_value_int;
    t->type_ids[total] = (uint8_t)type_id;

    switch (context->phase) {
      case IREE_TEMPLATE_PHASE_PREFIX:
        t->prefix_count++;
        break;
      case IREE_TEMPLATE_PHASE_INFIX:
        t->infix_count++;
        break;
      case IREE_TEMPLATE_PHASE_SUFFIX:
        t->suffix_count++;
        break;
    }
  }

  return iree_ok_status();
}

// Visitor callback for each piece in a template array.
static iree_status_t iree_tokenizer_template_piece_visitor(
    void* user_data, iree_host_size_t index, iree_string_view_t piece_value) {
  iree_template_compile_context_t* context =
      (iree_template_compile_context_t*)user_data;

  // Each piece is either {"Sequence":{"id":"A","type_id":0}} or
  // {"SpecialToken":{"id":"[CLS]","type_id":0}}.
  iree_string_view_t sequence_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_status_annotate_f(
      iree_json_try_lookup_object_value(piece_value, IREE_SV("Sequence"),
                                        &sequence_value),
      "parsing template piece at index %zu", index));

  if (sequence_value.size > 0) {
    // This is a Sequence piece.
    iree_string_view_t id_value = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
        sequence_value, IREE_SV("id"), &id_value));
    int64_t type_id = 0;
    iree_string_view_t type_id_value = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
        sequence_value, IREE_SV("type_id"), &type_id_value));
    IREE_RETURN_IF_ERROR(iree_json_parse_int64(type_id_value, &type_id));

    if (iree_string_view_equal(id_value, IREE_SV("A"))) {
      if (context->seen_sequence_a) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate Sequence A in template");
      }
      context->seen_sequence_a = true;
      context->template_out->sequence_a_type_id = (uint8_t)type_id;
      // After seeing A: next special tokens go to infix (for pair) or suffix.
      if (context->seen_sequence_b) {
        context->phase = IREE_TEMPLATE_PHASE_SUFFIX;
      } else {
        context->phase = IREE_TEMPLATE_PHASE_INFIX;
      }
    } else if (iree_string_view_equal(id_value, IREE_SV("B"))) {
      if (context->seen_sequence_b) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate Sequence B in template");
      }
      context->seen_sequence_b = true;
      context->template_out->sequence_b_type_id = (uint8_t)type_id;
      // After seeing B: next special tokens go to suffix.
      context->phase = IREE_TEMPLATE_PHASE_SUFFIX;
    } else {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "unknown Sequence id \"%.*s\" (expected \"A\" or \"B\")",
          (int)id_value.size, id_value.data);
    }
    return iree_ok_status();
  }

  // Try SpecialToken.
  iree_string_view_t special_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_status_annotate_f(
      iree_json_try_lookup_object_value(piece_value, IREE_SV("SpecialToken"),
                                        &special_value),
      "parsing template piece at index %zu", index));
  if (special_value.size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "template piece at index %zu is neither "
                            "Sequence nor SpecialToken",
                            index);
  }

  // Parse SpecialToken fields.
  iree_string_view_t token_id_str = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      special_value, IREE_SV("id"), &token_id_str));

  int64_t type_id = 0;
  iree_string_view_t type_id_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      special_value, IREE_SV("type_id"), &type_id_value));
  IREE_RETURN_IF_ERROR(iree_json_parse_int64(type_id_value, &type_id));

  return iree_tokenizer_template_emit_special_token(context, token_id_str,
                                                    (uint32_t)type_id);
}

// Compiles a single template (single or pair) piece array into a flat template.
static iree_status_t iree_tokenizer_compile_template_pieces(
    iree_string_view_t pieces_value, iree_string_view_t special_tokens_value,
    iree_tokenizer_postprocessor_template_t* out_template) {
  memset(out_template, 0, sizeof(*out_template));

  iree_template_compile_context_t context = {
      .template_out = out_template,
      .special_tokens_value = special_tokens_value,
      .phase = IREE_TEMPLATE_PHASE_PREFIX,
      .seen_sequence_a = false,
      .seen_sequence_b = false,
      .status = iree_ok_status(),
  };

  IREE_RETURN_IF_ERROR(iree_json_enumerate_array(
      pieces_value, iree_tokenizer_template_piece_visitor, &context));

  // For a single template (no B), everything after A is suffix, not infix.
  // The phase machine already handles this: if A is seen but not B, the phase
  // goes to INFIX but that's actually suffix. We need to fixup:
  if (context.seen_sequence_a && !context.seen_sequence_b &&
      out_template->infix_count > 0) {
    // Move infix to suffix (they were placed after A but no B exists).
    out_template->suffix_count = out_template->infix_count;
    out_template->infix_count = 0;
  }

  return iree_ok_status();
}

// JSON: {"type":"TemplateProcessing",
//        "single":[...pieces...],
//        "pair":[...pieces...],
//        "special_tokens":{...map...}}
static iree_status_t iree_tokenizer_parse_template_postprocessor(
    iree_string_view_t json,
    iree_tokenizer_postprocessor_t* out_postprocessor) {
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("single"),
      IREE_SVL("pair"),
      IREE_SVL("special_tokens"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  // All three fields are required for TemplateProcessing.
  iree_string_view_t single_value = iree_string_view_empty();
  iree_string_view_t pair_value = iree_string_view_empty();
  iree_string_view_t special_tokens_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_object_value(json, IREE_SV("single"), &single_value));
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_object_value(json, IREE_SV("pair"), &pair_value));
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      json, IREE_SV("special_tokens"), &special_tokens_value));

  // Compile single template.
  iree_tokenizer_postprocessor_template_t single_template = {0};
  IREE_RETURN_IF_ERROR(iree_status_annotate(
      iree_tokenizer_compile_template_pieces(single_value, special_tokens_value,
                                             &single_template),
      IREE_SV("in TemplateProcessing single template")));

  // Compile pair template.
  iree_tokenizer_postprocessor_template_t pair_template = {0};
  IREE_RETURN_IF_ERROR(iree_status_annotate(
      iree_tokenizer_compile_template_pieces(pair_value, special_tokens_value,
                                             &pair_template),
      IREE_SV("in TemplateProcessing pair template")));

  return iree_tokenizer_postprocessor_initialize(
      &single_template, &pair_template, IREE_TOKENIZER_POSTPROCESSOR_FLAG_NONE,
      out_postprocessor);
}

//===----------------------------------------------------------------------===//
// Sequence
//===----------------------------------------------------------------------===//

// Context for walking processors in a Sequence post-processor.
typedef struct iree_sequence_parse_context_t {
  iree_tokenizer_postprocessor_t* result;
  bool found_token_producer;
  bool found_byte_level;
  iree_tokenizer_postprocessor_flags_t flags;
  iree_status_t status;
} iree_sequence_parse_context_t;

// Visitor for each processor in a Sequence.
static iree_status_t iree_tokenizer_sequence_processor_visitor(
    void* user_data, iree_host_size_t index, iree_string_view_t value) {
  iree_sequence_parse_context_t* context =
      (iree_sequence_parse_context_t*)user_data;

  // Get the type of this sub-processor.
  iree_string_view_t type_str = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_object_value(value, IREE_SV("type"), &type_str));

  if (iree_string_view_equal(type_str, IREE_SV("ByteLevel"))) {
    // Extract trim_offsets and add_prefix_space (required, no serde default).
    context->found_byte_level = true;
    bool trim_offsets = false;
    bool add_prefix_space = false;
    IREE_RETURN_IF_ERROR(
        iree_json_lookup_bool(value, IREE_SV("trim_offsets"), &trim_offsets));
    IREE_RETURN_IF_ERROR(iree_json_lookup_bool(
        value, IREE_SV("add_prefix_space"), &add_prefix_space));
    if (trim_offsets) {
      context->flags |= IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS;
    }
    if (add_prefix_space) {
      context->flags |= IREE_TOKENIZER_POSTPROCESSOR_FLAG_ADD_PREFIX_SPACE;
    }
  } else if (!context->found_token_producer) {
    // First non-ByteLevel processor provides the template.
    context->found_token_producer = true;
    IREE_RETURN_IF_ERROR(iree_status_annotate_f(
        iree_tokenizer_huggingface_parse_postprocessor(value, context->result),
        "in Sequence processor at index %zu", index));
  }
  // Additional processors after the first token-producer are ignored
  // (matching HuggingFace behavior where later processors just pass through).

  return iree_ok_status();
}

// JSON: {"type":"Sequence","processors":[...]}
static iree_status_t iree_tokenizer_parse_sequence_postprocessor(
    iree_string_view_t json,
    iree_tokenizer_postprocessor_t* out_postprocessor) {
  static const iree_string_view_t kAllowedKeys[] = {
      IREE_SVL("type"),
      IREE_SVL("processors"),
  };
  IREE_RETURN_IF_ERROR(iree_json_validate_object_keys(
      json, kAllowedKeys, IREE_ARRAYSIZE(kAllowedKeys)));

  iree_string_view_t processors_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      json, IREE_SV("processors"), &processors_value));

  iree_sequence_parse_context_t context = {
      .result = out_postprocessor,
      .found_token_producer = false,
      .found_byte_level = false,
      .flags = IREE_TOKENIZER_POSTPROCESSOR_FLAG_NONE,
      .status = iree_ok_status(),
  };

  IREE_RETURN_IF_ERROR(iree_json_enumerate_array(
      processors_value, iree_tokenizer_sequence_processor_visitor, &context));

  // Merge ByteLevel flags into the result. If the token-producing child set
  // these flags already (e.g., RobertaProcessing), the ByteLevel flags take
  // precedence (OR semantics: if either says trim, we trim).
  if (context.found_byte_level) {
    out_postprocessor->flags |= context.flags;
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Top-Level Dispatch
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_huggingface_parse_postprocessor(
    iree_string_view_t postprocessor_value,
    iree_tokenizer_postprocessor_t* out_postprocessor) {
  IREE_ASSERT_ARGUMENT(out_postprocessor);
  memset(out_postprocessor, 0, sizeof(*out_postprocessor));

  // Handle null/empty (no post-processor configured).
  if (postprocessor_value.size == 0 ||
      iree_string_view_equal(postprocessor_value, IREE_SV("null"))) {
    return iree_ok_status();
  }

  // Get the type discriminator.
  iree_string_view_t type_str = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      postprocessor_value, IREE_SV("type"), &type_str));

  iree_status_t status;
  if (iree_string_view_equal(type_str, IREE_SV("BertProcessing"))) {
    status = iree_tokenizer_parse_bert_postprocessor(postprocessor_value,
                                                     out_postprocessor);
  } else if (iree_string_view_equal(type_str, IREE_SV("RobertaProcessing"))) {
    status = iree_tokenizer_parse_roberta_postprocessor(postprocessor_value,
                                                        out_postprocessor);
  } else if (iree_string_view_equal(type_str, IREE_SV("ByteLevel"))) {
    status = iree_tokenizer_parse_byte_level_postprocessor(postprocessor_value,
                                                           out_postprocessor);
  } else if (iree_string_view_equal(type_str, IREE_SV("TemplateProcessing"))) {
    status = iree_tokenizer_parse_template_postprocessor(postprocessor_value,
                                                         out_postprocessor);
  } else if (iree_string_view_equal(type_str, IREE_SV("Sequence"))) {
    status = iree_tokenizer_parse_sequence_postprocessor(postprocessor_value,
                                                         out_postprocessor);
  } else {
    status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported post_processor type \"%.*s\"",
                              (int)type_str.size, type_str.data);
  }

  if (!iree_status_is_ok(status)) {
    status =
        iree_status_annotate_f(status, "parsing post_processor type \"%.*s\"",
                               (int)type_str.size, type_str.data);
  }
  return status;
}
