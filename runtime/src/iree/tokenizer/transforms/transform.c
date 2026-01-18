// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/transforms/transform.h"

#include "iree/tokenizer/transforms/bert.h"
#include "iree/tokenizer/transforms/byte_level.h"
#include "iree/tokenizer/transforms/metaspace.h"
#include "iree/tokenizer/transforms/sequence.h"
#include "iree/tokenizer/transforms/split.h"

//===----------------------------------------------------------------------===//
// Transform Lifecycle
//===----------------------------------------------------------------------===//

void iree_tokenizer_text_transform_deinitialize(
    iree_tokenizer_text_transform_t* transform) {
  if (!transform) return;

  // Deinitialize embedded normalizer (may have sequence children).
  iree_tokenizer_normalizer_deinitialize(&transform->normalizer);

  switch (transform->type) {
    case IREE_TOKENIZER_TEXT_TRANSFORM_SEQUENCE: {
      iree_tokenizer_sequence_config_t* config = &transform->config.sequence;
      if (config->children) {
        for (iree_host_size_t i = 0; i < config->count; ++i) {
          iree_tokenizer_text_transform_deinitialize(&config->children[i]);
        }
        iree_allocator_free(config->allocator, config->children);
        config->children = NULL;
        config->count = 0;
      }
      break;
    }
    case IREE_TOKENIZER_TEXT_TRANSFORM_SPLIT:
      iree_tokenizer_split_config_deinitialize(&transform->config.split);
      break;
    default:
      break;
  }
  memset(transform, 0, sizeof(*transform));
}

iree_status_t iree_tokenizer_text_transform_initialize_sequence(
    iree_tokenizer_text_transform_t* children, iree_host_size_t count,
    iree_allocator_t allocator,
    iree_tokenizer_text_transform_t* out_transform) {
  IREE_ASSERT_ARGUMENT(out_transform);
  memset(out_transform, 0, sizeof(*out_transform));
  iree_tokenizer_normalizer_initialize_none(&out_transform->normalizer);

  if (count == 0) {
    out_transform->type = IREE_TOKENIZER_TEXT_TRANSFORM_NONE;
    return iree_ok_status();
  }

  IREE_ASSERT_ARGUMENT(children);

  // Check for overflow in children array size.
  if (count > IREE_HOST_SIZE_MAX / sizeof(iree_tokenizer_text_transform_t)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "sequence child count overflow");
  }
  iree_host_size_t children_size =
      count * sizeof(iree_tokenizer_text_transform_t);

  // Allocate children array and move children into it.
  iree_tokenizer_text_transform_t* owned_children = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, children_size, (void**)&owned_children));

  // Move children: copy structs and zero originals to transfer ownership.
  memcpy(owned_children, children, children_size);
  memset(children, 0, children_size);

  out_transform->type = IREE_TOKENIZER_TEXT_TRANSFORM_SEQUENCE;
  out_transform->config.sequence.count = count;
  out_transform->config.sequence.children = owned_children;
  out_transform->config.sequence.allocator = allocator;

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Encode Dispatch
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_text_transform_encode(
    const iree_tokenizer_normalizer_t* normalizer,
    const iree_tokenizer_text_transform_t* transform, iree_string_view_t text,
    iree_tokenizer_string_callback_fn_t callback, void* user_data) {
  IREE_ASSERT_ARGUMENT(transform);
  IREE_ASSERT_ARGUMENT(callback);

  // Use provided normalizer, or fall back to transform's embedded normalizer.
  const iree_tokenizer_normalizer_t* effective_normalizer =
      normalizer ? normalizer : &transform->normalizer;

  switch (transform->type) {
    case IREE_TOKENIZER_TEXT_TRANSFORM_NONE: {
      // Passthrough: apply normalizer (if any), emit entire text as segment.
      if (text.size == 0) return iree_ok_status();

      // Apply normalizer if present.
      if (effective_normalizer->type != IREE_TOKENIZER_NORMALIZER_NONE) {
        char normalized_buffer[8192];
        iree_host_size_t normalized_length = 0;
        IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_apply(
            effective_normalizer, text, normalized_buffer,
            sizeof(normalized_buffer), &normalized_length));
        iree_string_view_t normalized =
            iree_make_string_view(normalized_buffer, normalized_length);
        iree_string_view_list_t list = {1, &normalized};
        return callback(user_data, list);
      }

      iree_string_view_list_t list = {1, &text};
      return callback(user_data, list);
    }

    case IREE_TOKENIZER_TEXT_TRANSFORM_BERT:
      return iree_tokenizer_bert_encode(effective_normalizer, text, callback,
                                        user_data);

    case IREE_TOKENIZER_TEXT_TRANSFORM_WHITESPACE:
      return iree_tokenizer_whitespace_encode(effective_normalizer, text,
                                              callback, user_data);

    case IREE_TOKENIZER_TEXT_TRANSFORM_METASPACE:
      return iree_tokenizer_metaspace_encode(effective_normalizer,
                                             &transform->config.metaspace, text,
                                             callback, user_data);

    case IREE_TOKENIZER_TEXT_TRANSFORM_BYTE_LEVEL:
      return iree_tokenizer_byte_level_encode(effective_normalizer,
                                              &transform->config.byte_level,
                                              text, callback, user_data);

    case IREE_TOKENIZER_TEXT_TRANSFORM_SEQUENCE:
      return iree_tokenizer_sequence_encode(effective_normalizer,
                                            &transform->config.sequence, text,
                                            callback, user_data);

    case IREE_TOKENIZER_TEXT_TRANSFORM_SPLIT:
      return iree_tokenizer_split_encode(effective_normalizer,
                                         &transform->config.split, text,
                                         callback, user_data);

    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown text transform type %d",
                              (int)transform->type);
  }
}

//===----------------------------------------------------------------------===//
// Decode Dispatch
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_text_transform_decode(
    const iree_tokenizer_text_transform_t* transform, iree_string_view_t text,
    char* out_buffer, iree_host_size_t max_size, iree_host_size_t* out_size) {
  IREE_ASSERT_ARGUMENT(transform);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_ASSERT_ARGUMENT(out_size);
  *out_size = 0;

  switch (transform->type) {
    case IREE_TOKENIZER_TEXT_TRANSFORM_NONE:
    case IREE_TOKENIZER_TEXT_TRANSFORM_BERT:
    case IREE_TOKENIZER_TEXT_TRANSFORM_WHITESPACE:
      // Passthrough: copy text to output (handles overlapping buffers).
      if (text.size > max_size) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "output buffer too small");
      }
      memmove(out_buffer, text.data, text.size);
      *out_size = text.size;
      return iree_ok_status();

    case IREE_TOKENIZER_TEXT_TRANSFORM_METASPACE:
      return iree_tokenizer_metaspace_decode(&transform->config.metaspace, text,
                                             out_buffer, max_size, out_size);

    case IREE_TOKENIZER_TEXT_TRANSFORM_BYTE_LEVEL:
      return iree_tokenizer_byte_level_decode(
          &transform->config.byte_level, text, out_buffer, max_size, out_size);

    case IREE_TOKENIZER_TEXT_TRANSFORM_SEQUENCE:
      return iree_tokenizer_sequence_decode(&transform->config.sequence, text,
                                            out_buffer, max_size, out_size);

    case IREE_TOKENIZER_TEXT_TRANSFORM_SPLIT:
      return iree_tokenizer_split_decode(&transform->config.split, text,
                                         out_buffer, max_size, out_size);

    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown text transform type %d",
                              (int)transform->type);
  }
}
