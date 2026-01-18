// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/transforms/split.h"

#include "iree/tokenizer/regex/compile.h"

//===----------------------------------------------------------------------===//
// Split Config Lifecycle
//===----------------------------------------------------------------------===//

void iree_tokenizer_split_config_deinitialize(
    iree_tokenizer_split_config_t* config) {
  if (config->pattern_storage) {
    iree_tokenizer_regex_compiled_free(config->pattern_storage,
                                       config->allocator);
    config->pattern_storage = NULL;
  }
  memset(&config->dfa, 0, sizeof(config->dfa));
}

//===----------------------------------------------------------------------===//
// Split Transform Initializer
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_text_transform_initialize_split(
    iree_string_view_t pattern, iree_tokenizer_regex_split_behavior_t behavior,
    bool invert, iree_allocator_t allocator,
    iree_tokenizer_text_transform_t* out_transform) {
  IREE_ASSERT_ARGUMENT(out_transform);
  memset(out_transform, 0, sizeof(*out_transform));

  // Validate behavior enum range.
  if (behavior < IREE_TOKENIZER_REGEX_SPLIT_REMOVED ||
      behavior > IREE_TOKENIZER_REGEX_SPLIT_CONTIGUOUS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid split behavior %d", (int)behavior);
  }

  out_transform->type = IREE_TOKENIZER_TEXT_TRANSFORM_SPLIT;
  iree_tokenizer_normalizer_initialize_none(&out_transform->normalizer);

  iree_tokenizer_split_config_t* config = &out_transform->config.split;
  config->behavior = behavior;
  config->invert = invert;
  config->allocator = allocator;

  // Compile the regex pattern to a DFA.
  iree_tokenizer_regex_compile_error_t compile_error = {0};
  iree_status_t status = iree_tokenizer_regex_compile_and_load(
      pattern, IREE_TOKENIZER_REGEX_COMPILE_FLAG_NONE, allocator, &config->dfa,
      &config->pattern_storage, &compile_error);
  if (!iree_status_is_ok(status)) {
    iree_tokenizer_text_transform_initialize_none(out_transform);
    return status;
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Split Encode - Streaming Implementation
//===----------------------------------------------------------------------===//

// Streaming context for processing matches as they arrive from regex executor.
// This enables unlimited input length by processing one match at a time.
typedef struct {
  // Input text being processed.
  iree_string_view_t text;
  // Current position in text (start of next gap).
  iree_host_size_t position;
  // Split behavior configuration.
  iree_tokenizer_regex_split_behavior_t behavior;
  bool invert;

  // Pending match for behaviors that need lookahead (MERGED_WITH_NEXT,
  // CONTIGUOUS). When has_pending is true, we hold this match until we see
  // the next match (or end of input) to know where the segment ends.
  bool has_pending;
  iree_tokenizer_regex_match_t pending_match;
  // For CONTIGUOUS: end of current contiguous run being accumulated.
  iree_host_size_t contiguous_run_end;

  // Segment emission batching.
  iree_string_view_t segments[IREE_TOKENIZER_STRING_BATCH_CAPACITY];
  iree_host_size_t segment_count;
  iree_tokenizer_string_callback_fn_t callback;
  void* user_data;
} iree_tokenizer_split_stream_context_t;

// Emit a segment, flushing the batch if full.
static iree_status_t iree_tokenizer_split_stream_emit(
    iree_tokenizer_split_stream_context_t* context,
    iree_string_view_t segment) {
  if (segment.size == 0) return iree_ok_status();

  if (context->segment_count >= IREE_TOKENIZER_STRING_BATCH_CAPACITY) {
    iree_string_view_list_t list = {context->segment_count, context->segments};
    IREE_RETURN_IF_ERROR(context->callback(context->user_data, list));
    context->segment_count = 0;
  }
  context->segments[context->segment_count++] = segment;
  return iree_ok_status();
}

// Flush any remaining segments.
static iree_status_t iree_tokenizer_split_stream_flush(
    iree_tokenizer_split_stream_context_t* context) {
  if (context->segment_count == 0) return iree_ok_status();
  iree_string_view_list_t list = {context->segment_count, context->segments};
  IREE_RETURN_IF_ERROR(context->callback(context->user_data, list));
  context->segment_count = 0;
  return iree_ok_status();
}

// Process a single match in streaming fashion.
static iree_status_t iree_tokenizer_split_stream_process_match(
    void* user_data, iree_tokenizer_regex_match_t match) {
  iree_tokenizer_split_stream_context_t* context =
      (iree_tokenizer_split_stream_context_t*)user_data;

  if (context->invert) {
    // INVERTED: just emit the match, ignore gaps.
    iree_string_view_t segment = iree_make_string_view(
        context->text.data + match.start, match.end - match.start);
    return iree_tokenizer_split_stream_emit(context, segment);
  }

  switch ((iree_tokenizer_regex_split_behavior_t)context->behavior) {
    case IREE_TOKENIZER_REGEX_SPLIT_REMOVED: {
      // Emit gap before match, skip the match itself.
      iree_string_view_t gap =
          iree_make_string_view(context->text.data + context->position,
                                match.start - context->position);
      IREE_RETURN_IF_ERROR(iree_tokenizer_split_stream_emit(context, gap));
      context->position = match.end;
      break;
    }

    case IREE_TOKENIZER_REGEX_SPLIT_ISOLATED: {
      // Emit gap, then emit match as separate segment.
      iree_string_view_t gap =
          iree_make_string_view(context->text.data + context->position,
                                match.start - context->position);
      IREE_RETURN_IF_ERROR(iree_tokenizer_split_stream_emit(context, gap));
      iree_string_view_t delimiter = iree_make_string_view(
          context->text.data + match.start, match.end - match.start);
      IREE_RETURN_IF_ERROR(
          iree_tokenizer_split_stream_emit(context, delimiter));
      context->position = match.end;
      break;
    }

    case IREE_TOKENIZER_REGEX_SPLIT_MERGED_WITH_PREVIOUS: {
      // Emit (gap + match) as one segment.
      iree_string_view_t merged =
          iree_make_string_view(context->text.data + context->position,
                                match.end - context->position);
      IREE_RETURN_IF_ERROR(iree_tokenizer_split_stream_emit(context, merged));
      context->position = match.end;
      break;
    }

    case IREE_TOKENIZER_REGEX_SPLIT_MERGED_WITH_NEXT: {
      // Need lookahead: hold this match, emit previous pending match now that
      // we know where it ends.
      if (context->has_pending) {
        // Emit pending match + gap up to current match.
        iree_string_view_t segment = iree_make_string_view(
            context->text.data + context->pending_match.start,
            match.start - context->pending_match.start);
        IREE_RETURN_IF_ERROR(
            iree_tokenizer_split_stream_emit(context, segment));
      } else {
        // First match: emit leading text before it.
        if (match.start > 0) {
          iree_string_view_t leading =
              iree_make_string_view(context->text.data, match.start);
          IREE_RETURN_IF_ERROR(
              iree_tokenizer_split_stream_emit(context, leading));
        }
      }
      context->pending_match = match;
      context->has_pending = true;
      context->position = match.end;
      break;
    }

    case IREE_TOKENIZER_REGEX_SPLIT_CONTIGUOUS: {
      // Check if this match is contiguous with the current run.
      if (context->has_pending && match.start == context->contiguous_run_end) {
        // Extend the contiguous run.
        context->contiguous_run_end = match.end;
      } else {
        // Not contiguous - emit pending run if any.
        if (context->has_pending) {
          iree_string_view_t run = iree_make_string_view(
              context->text.data + context->pending_match.start,
              context->contiguous_run_end - context->pending_match.start);
          IREE_RETURN_IF_ERROR(iree_tokenizer_split_stream_emit(context, run));
        }
        // Emit gap before this new match.
        iree_string_view_t gap =
            iree_make_string_view(context->text.data + context->position,
                                  match.start - context->position);
        IREE_RETURN_IF_ERROR(iree_tokenizer_split_stream_emit(context, gap));
        // Start new contiguous run.
        context->pending_match = match;
        context->contiguous_run_end = match.end;
        context->has_pending = true;
      }
      context->position = match.end;
      break;
    }

    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown split behavior %u", context->behavior);
  }

  return iree_ok_status();
}

// Finalize streaming: emit any pending state and trailing text.
static iree_status_t iree_tokenizer_split_stream_finalize(
    iree_tokenizer_split_stream_context_t* context) {
  if (context->invert) {
    // INVERTED: nothing to finalize (gaps are discarded).
    return iree_ok_status();
  }

  switch ((iree_tokenizer_regex_split_behavior_t)context->behavior) {
    case IREE_TOKENIZER_REGEX_SPLIT_REMOVED:
    case IREE_TOKENIZER_REGEX_SPLIT_ISOLATED:
    case IREE_TOKENIZER_REGEX_SPLIT_MERGED_WITH_PREVIOUS: {
      // Emit trailing text after last match.
      if (context->position < context->text.size) {
        iree_string_view_t trailing =
            iree_make_string_view(context->text.data + context->position,
                                  context->text.size - context->position);
        IREE_RETURN_IF_ERROR(
            iree_tokenizer_split_stream_emit(context, trailing));
      }
      break;
    }

    case IREE_TOKENIZER_REGEX_SPLIT_MERGED_WITH_NEXT: {
      if (context->has_pending) {
        // Emit pending match + trailing text to end.
        iree_string_view_t segment = iree_make_string_view(
            context->text.data + context->pending_match.start,
            context->text.size - context->pending_match.start);
        IREE_RETURN_IF_ERROR(
            iree_tokenizer_split_stream_emit(context, segment));
      } else {
        // No matches at all - emit entire text.
        IREE_RETURN_IF_ERROR(
            iree_tokenizer_split_stream_emit(context, context->text));
      }
      break;
    }

    case IREE_TOKENIZER_REGEX_SPLIT_CONTIGUOUS: {
      // Emit pending contiguous run if any.
      if (context->has_pending) {
        iree_string_view_t run = iree_make_string_view(
            context->text.data + context->pending_match.start,
            context->contiguous_run_end - context->pending_match.start);
        IREE_RETURN_IF_ERROR(iree_tokenizer_split_stream_emit(context, run));
      }
      // Emit trailing text.
      if (context->position < context->text.size) {
        iree_string_view_t trailing =
            iree_make_string_view(context->text.data + context->position,
                                  context->text.size - context->position);
        IREE_RETURN_IF_ERROR(
            iree_tokenizer_split_stream_emit(context, trailing));
      }
      break;
    }

    default:
      break;
  }

  return iree_ok_status();
}

iree_status_t iree_tokenizer_split_encode(
    const iree_tokenizer_normalizer_t* normalizer,
    const iree_tokenizer_split_config_t* config, iree_string_view_t text,
    iree_tokenizer_string_callback_fn_t callback, void* user_data) {
  IREE_ASSERT_ARGUMENT(config);

  if (text.size == 0) return iree_ok_status();

  // Pre-normalize the entire text before splitting. Many normalizers require
  // string-level processing (NFC for canonical reordering, Sequence for
  // chaining multiple normalizers, etc.).
  char normalized_stack_buffer[8192];
  iree_string_view_t normalized_text = text;
  if (normalizer != NULL &&
      normalizer->type != IREE_TOKENIZER_NORMALIZER_NONE) {
    iree_host_size_t normalized_length = 0;
    IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_apply(
        normalizer, text, normalized_stack_buffer,
        sizeof(normalized_stack_buffer), &normalized_length));
    normalized_text =
        iree_make_string_view(normalized_stack_buffer, normalized_length);
  }

  // For offset-tracking behaviors (MergedWith*), segments are NOT BPE word
  // boundaries. The BPE model should see the full text as one unit - the split
  // is only metadata for character-to-token offset mapping. Emit the entire
  // normalized text as a single segment.
  if (!iree_tokenizer_regex_split_creates_bpe_boundaries(
          (iree_tokenizer_regex_split_behavior_t)config->behavior)) {
    iree_string_view_list_t list = {1, &normalized_text};
    return callback(user_data, list);
  }

  // Initialize streaming context.
  iree_tokenizer_split_stream_context_t stream_context = {
      .text = normalized_text,
      .position = 0,
      .behavior = config->behavior,
      .invert = config->invert,
      .has_pending = false,
      .pending_match = {0, 0},
      .contiguous_run_end = 0,
      .segment_count = 0,
      .callback = callback,
      .user_data = user_data,
  };

  // Execute regex with streaming callback - processes unlimited matches.
  IREE_RETURN_IF_ERROR(iree_tokenizer_regex_exec(
      &config->dfa, normalized_text, iree_tokenizer_split_stream_process_match,
      &stream_context));

  // Finalize: emit pending state and trailing text.
  IREE_RETURN_IF_ERROR(iree_tokenizer_split_stream_finalize(&stream_context));

  // Flush any remaining buffered segments.
  return iree_tokenizer_split_stream_flush(&stream_context);
}

//===----------------------------------------------------------------------===//
// Split Decode
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_split_decode(
    const iree_tokenizer_split_config_t* config, iree_string_view_t text,
    char* out_buffer, iree_host_size_t max_size, iree_host_size_t* out_size) {
  IREE_ASSERT_ARGUMENT(config);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_ASSERT_ARGUMENT(out_size);

  // Split is a passthrough for decode - just copy input to output.
  if (text.size > max_size) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "output buffer too small");
  }
  memmove(out_buffer, text.data, text.size);
  *out_size = text.size;
  return iree_ok_status();
}
