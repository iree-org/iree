// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/postprocessor.h"

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_postprocessor_initialize_template(
    iree_tokenizer_template_piece_t* templates, iree_host_size_t single_count,
    iree_host_size_t pair_count, iree_allocator_t allocator,
    iree_tokenizer_postprocessor_t* out_pp) {
  memset(out_pp, 0, sizeof(*out_pp));
  out_pp->type = IREE_TOKENIZER_POSTPROCESSOR_TEMPLATE;
  out_pp->config.template_.templates = templates;
  out_pp->config.template_.single_count = single_count;
  out_pp->config.template_.pair_count = pair_count;
  out_pp->config.template_.allocator = allocator;
  return iree_ok_status();
}

iree_status_t iree_tokenizer_postprocessor_initialize_sequence(
    iree_tokenizer_postprocessor_t* children, iree_host_size_t count,
    iree_allocator_t allocator, iree_tokenizer_postprocessor_t* out_pp) {
  memset(out_pp, 0, sizeof(*out_pp));
  out_pp->type = IREE_TOKENIZER_POSTPROCESSOR_SEQUENCE;
  out_pp->config.sequence.children = children;
  out_pp->config.sequence.count = count;
  out_pp->config.sequence.allocator = allocator;
  return iree_ok_status();
}

void iree_tokenizer_postprocessor_deinitialize(
    iree_tokenizer_postprocessor_t* pp) {
  if (!pp) return;
  switch (pp->type) {
    case IREE_TOKENIZER_POSTPROCESSOR_TEMPLATE:
      if (pp->config.template_.templates) {
        iree_allocator_free(pp->config.template_.allocator,
                            pp->config.template_.templates);
      }
      break;
    case IREE_TOKENIZER_POSTPROCESSOR_SEQUENCE:
      // Recursively deinitialize children.
      for (iree_host_size_t i = 0; i < pp->config.sequence.count; ++i) {
        iree_tokenizer_postprocessor_deinitialize(
            &pp->config.sequence.children[i]);
      }
      if (pp->config.sequence.children) {
        iree_allocator_free(pp->config.sequence.allocator,
                            pp->config.sequence.children);
      }
      break;
    default:
      // NONE, ROBERTA, BYTE_LEVEL have no heap allocations.
      break;
  }
  memset(pp, 0, sizeof(*pp));
}

//===----------------------------------------------------------------------===//
// Apply Template
//===----------------------------------------------------------------------===//

// Applies a template (single or pair) to the input text(s).
static iree_status_t iree_tokenizer_postprocessor_apply_template(
    const iree_tokenizer_template_piece_t* pieces, iree_host_size_t piece_count,
    iree_string_view_t text_a, iree_string_view_t text_b,
    iree_tokenizer_encode_text_fn_t encode_text_fn,
    iree_tokenizer_emit_token_fn_t emit_token_fn, void* user_data) {
  for (iree_host_size_t i = 0; i < piece_count; ++i) {
    const iree_tokenizer_template_piece_t* piece = &pieces[i];
    switch (piece->type) {
      case IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_A: {
        IREE_RETURN_IF_ERROR(encode_text_fn(user_data, text_a));
        break;
      }
      case IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_B: {
        IREE_RETURN_IF_ERROR(encode_text_fn(user_data, text_b));
        break;
      }
      case IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL: {
        IREE_RETURN_IF_ERROR(emit_token_fn(user_data, piece->token_id));
        break;
      }
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Apply Single
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_postprocessor_apply_single(
    const iree_tokenizer_postprocessor_t* pp, iree_string_view_t text,
    iree_tokenizer_encode_text_fn_t encode_text_fn,
    iree_tokenizer_emit_token_fn_t emit_token_fn, void* user_data) {
  if (!pp) {
    // NULL post-processor means no special tokens, just encode text.
    return encode_text_fn(user_data, text);
  }

  switch (pp->type) {
    case IREE_TOKENIZER_POSTPROCESSOR_NONE:
      // No special tokens - just encode the text.
      return encode_text_fn(user_data, text);

    case IREE_TOKENIZER_POSTPROCESSOR_TEMPLATE: {
      if (pp->config.template_.single_count == 0) {
        // Empty template - just encode text.
        return encode_text_fn(user_data, text);
      }
      return iree_tokenizer_postprocessor_apply_template(
          pp->config.template_.templates, pp->config.template_.single_count,
          text, iree_string_view_empty(), encode_text_fn, emit_token_fn,
          user_data);
    }

    case IREE_TOKENIZER_POSTPROCESSOR_ROBERTA: {
      // RoBERTa format: <s> $A </s>
      IREE_RETURN_IF_ERROR(emit_token_fn(user_data, pp->config.roberta.cls_id));
      IREE_RETURN_IF_ERROR(encode_text_fn(user_data, text));
      IREE_RETURN_IF_ERROR(emit_token_fn(user_data, pp->config.roberta.sep_id));
      return iree_ok_status();
    }

    case IREE_TOKENIZER_POSTPROCESSOR_BYTE_LEVEL:
      // ByteLevel in post-processor context is typically a no-op for encoding.
      // The actual byte-level handling is in the pre-tokenizer/decoder.
      return encode_text_fn(user_data, text);

    case IREE_TOKENIZER_POSTPROCESSOR_SEQUENCE:
      // HuggingFace Sequence post-processors chain multiple processors, but in
      // practice only one child adds special tokens (TemplateProcessing or
      // RoBERTa). Other children like ByteLevel are no-ops for encoding.
      //
      // Assumption: Only ONE token-adding child exists in the Sequence. This
      // matches all real-world HuggingFace tokenizers (e.g., Llama 3 chains
      // ByteLevel + TemplateProcessing, where ByteLevel is a no-op here).
      // If multiple token-adding children existed, prefix/suffix emission could
      // become inconsistent - but no known models require this.
      for (iree_host_size_t i = 0; i < pp->config.sequence.count; ++i) {
        const iree_tokenizer_postprocessor_t* child =
            &pp->config.sequence.children[i];
        if (child->type == IREE_TOKENIZER_POSTPROCESSOR_TEMPLATE ||
            child->type == IREE_TOKENIZER_POSTPROCESSOR_ROBERTA) {
          return iree_tokenizer_postprocessor_apply_single(
              child, text, encode_text_fn, emit_token_fn, user_data);
        }
      }
      // No template-based processor found in chain, just encode text.
      return encode_text_fn(user_data, text);
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Apply Pair
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_postprocessor_apply_pair(
    const iree_tokenizer_postprocessor_t* pp, iree_string_view_t text_a,
    iree_string_view_t text_b, iree_tokenizer_encode_text_fn_t encode_text_fn,
    iree_tokenizer_emit_token_fn_t emit_token_fn, void* user_data) {
  if (!pp) {
    // NULL post-processor means no special tokens.
    IREE_RETURN_IF_ERROR(encode_text_fn(user_data, text_a));
    return encode_text_fn(user_data, text_b);
  }

  switch (pp->type) {
    case IREE_TOKENIZER_POSTPROCESSOR_NONE: {
      // No special tokens - just encode both texts.
      IREE_RETURN_IF_ERROR(encode_text_fn(user_data, text_a));
      return encode_text_fn(user_data, text_b);
    }

    case IREE_TOKENIZER_POSTPROCESSOR_TEMPLATE: {
      if (pp->config.template_.pair_count == 0) {
        // No pair template defined - encode both texts directly without
        // template processing.
        IREE_RETURN_IF_ERROR(encode_text_fn(user_data, text_a));
        return encode_text_fn(user_data, text_b);
      }
      // Pair template starts after single template pieces.
      const iree_tokenizer_template_piece_t* pair_pieces =
          pp->config.template_.templates + pp->config.template_.single_count;
      return iree_tokenizer_postprocessor_apply_template(
          pair_pieces, pp->config.template_.pair_count, text_a, text_b,
          encode_text_fn, emit_token_fn, user_data);
    }

    case IREE_TOKENIZER_POSTPROCESSOR_ROBERTA: {
      // RoBERTa pair format: <s> $A </s></s> $B </s>
      IREE_RETURN_IF_ERROR(emit_token_fn(user_data, pp->config.roberta.cls_id));
      IREE_RETURN_IF_ERROR(encode_text_fn(user_data, text_a));
      IREE_RETURN_IF_ERROR(emit_token_fn(user_data, pp->config.roberta.sep_id));
      IREE_RETURN_IF_ERROR(emit_token_fn(user_data, pp->config.roberta.sep_id));
      IREE_RETURN_IF_ERROR(encode_text_fn(user_data, text_b));
      IREE_RETURN_IF_ERROR(emit_token_fn(user_data, pp->config.roberta.sep_id));
      return iree_ok_status();
    }

    case IREE_TOKENIZER_POSTPROCESSOR_BYTE_LEVEL: {
      // ByteLevel is typically a no-op for pair encoding.
      IREE_RETURN_IF_ERROR(encode_text_fn(user_data, text_a));
      return encode_text_fn(user_data, text_b);
    }

    case IREE_TOKENIZER_POSTPROCESSOR_SEQUENCE: {
      // Find the first token-adding child and use it.
      // Assumption: Only ONE token-adding child exists (see apply_single).
      for (iree_host_size_t i = 0; i < pp->config.sequence.count; ++i) {
        const iree_tokenizer_postprocessor_t* child =
            &pp->config.sequence.children[i];
        if (child->type == IREE_TOKENIZER_POSTPROCESSOR_TEMPLATE ||
            child->type == IREE_TOKENIZER_POSTPROCESSOR_ROBERTA) {
          return iree_tokenizer_postprocessor_apply_pair(
              child, text_a, text_b, encode_text_fn, emit_token_fn, user_data);
        }
      }
      // No template-based processor found.
      IREE_RETURN_IF_ERROR(encode_text_fn(user_data, text_a));
      return encode_text_fn(user_data, text_b);
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Special Token Emission
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_postprocessor_emit_prefix(
    const iree_tokenizer_postprocessor_t* pp,
    iree_tokenizer_emit_token_fn_t emit_token_fn, void* user_data) {
  if (!pp) return iree_ok_status();

  switch (pp->type) {
    case IREE_TOKENIZER_POSTPROCESSOR_NONE:
    case IREE_TOKENIZER_POSTPROCESSOR_BYTE_LEVEL:
      // No prefix tokens.
      return iree_ok_status();

    case IREE_TOKENIZER_POSTPROCESSOR_TEMPLATE: {
      // Emit all SPECIAL tokens before SEQUENCE_A in single template.
      const iree_tokenizer_template_piece_t* pieces =
          pp->config.template_.templates;
      for (iree_host_size_t i = 0; i < pp->config.template_.single_count; ++i) {
        if (pieces[i].type == IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_A) {
          break;  // Reached $A - stop emitting prefix.
        }
        if (pieces[i].type == IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL) {
          IREE_RETURN_IF_ERROR(emit_token_fn(user_data, pieces[i].token_id));
        }
      }
      return iree_ok_status();
    }

    case IREE_TOKENIZER_POSTPROCESSOR_ROBERTA:
      // RoBERTa prefix is just <s> (cls_id).
      return emit_token_fn(user_data, pp->config.roberta.cls_id);

    case IREE_TOKENIZER_POSTPROCESSOR_SEQUENCE:
      // Emit prefix from the first token-adding child.
      // Assumption: Only ONE token-adding child exists (see apply_single).
      for (iree_host_size_t i = 0; i < pp->config.sequence.count; ++i) {
        const iree_tokenizer_postprocessor_t* child =
            &pp->config.sequence.children[i];
        if (child->type == IREE_TOKENIZER_POSTPROCESSOR_TEMPLATE ||
            child->type == IREE_TOKENIZER_POSTPROCESSOR_ROBERTA) {
          return iree_tokenizer_postprocessor_emit_prefix(child, emit_token_fn,
                                                          user_data);
        }
      }
      return iree_ok_status();
  }

  return iree_ok_status();
}

iree_status_t iree_tokenizer_postprocessor_emit_suffix(
    const iree_tokenizer_postprocessor_t* pp,
    iree_tokenizer_emit_token_fn_t emit_token_fn, void* user_data) {
  if (!pp) return iree_ok_status();

  switch (pp->type) {
    case IREE_TOKENIZER_POSTPROCESSOR_NONE:
    case IREE_TOKENIZER_POSTPROCESSOR_BYTE_LEVEL:
      // No suffix tokens.
      return iree_ok_status();

    case IREE_TOKENIZER_POSTPROCESSOR_TEMPLATE: {
      // Emit all SPECIAL tokens after SEQUENCE_A in single template.
      const iree_tokenizer_template_piece_t* pieces =
          pp->config.template_.templates;
      bool found_sequence_a = false;
      for (iree_host_size_t i = 0; i < pp->config.template_.single_count; ++i) {
        if (pieces[i].type == IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_A) {
          found_sequence_a = true;
        } else if (found_sequence_a &&
                   pieces[i].type == IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL) {
          IREE_RETURN_IF_ERROR(emit_token_fn(user_data, pieces[i].token_id));
        }
      }
      return iree_ok_status();
    }

    case IREE_TOKENIZER_POSTPROCESSOR_ROBERTA:
      // RoBERTa suffix is just </s> (sep_id).
      return emit_token_fn(user_data, pp->config.roberta.sep_id);

    case IREE_TOKENIZER_POSTPROCESSOR_SEQUENCE:
      // Emit suffix from the last token-adding child (searches backwards).
      // Assumption: Only ONE token-adding child exists (see apply_single), so
      // the search direction doesn't matter in practice.
      for (iree_host_size_t i = pp->config.sequence.count; i > 0; --i) {
        const iree_tokenizer_postprocessor_t* child =
            &pp->config.sequence.children[i - 1];
        if (child->type == IREE_TOKENIZER_POSTPROCESSOR_TEMPLATE ||
            child->type == IREE_TOKENIZER_POSTPROCESSOR_ROBERTA) {
          return iree_tokenizer_postprocessor_emit_suffix(child, emit_token_fn,
                                                          user_data);
        }
      }
      return iree_ok_status();
  }

  return iree_ok_status();
}
