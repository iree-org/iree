// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./tokenizer.h"

#include <vector>

#include "./status_utils.h"
#include "iree/io/file_contents.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/format/tiktoken/tiktoken.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/tokenizer/vocab/vocab.h"

namespace iree::python {

// GIL release pattern: All C tokenizer API calls are wrapped in
// py::gil_scoped_release blocks to allow other Python threads to run during
// tokenization. IMPORTANT: CheckApiStatus (which calls PyErr_SetString) must
// be called AFTER the GIL is re-acquired — never inside the release block.
// Capture the iree_status_t, close the release scope, then check.

namespace {

// Output buffer size for streaming and batch decode. Derived from the C API's
// recommended size, which was benchmarked to achieve full decode throughput.
static constexpr iree_host_size_t kDecodeBufferSize =
    IREE_TOKENIZER_DECODE_OUTPUT_RECOMMENDED_SIZE;

// Token output buffer size for streaming encode feed calls. Chosen to amortize
// per-call overhead (GIL release/acquire, list append) while keeping the
// allocation small. The feed loop handles partial consumption regardless of
// this value.
static constexpr iree_host_size_t kStreamTokenCapacity = 256;

// Expected average chunk size hint for the streaming transform buffer.
// The C API uses this to size internal ring buffers for the
// normalizer/segmenter pipeline. Larger values waste memory, smaller values
// cause more frequent ring buffer wraps. 4KB covers typical chat prompts.
static constexpr iree_host_size_t kStreamChunkSizeHint = 4096;

}  // namespace

// ============================================================================
// Tokenizer
// ============================================================================

class EncodeStream;
class DecodeStream;

class Tokenizer {
 public:
  Tokenizer() : tokenizer_(nullptr) {}
  Tokenizer(Tokenizer&& other) : tokenizer_(other.tokenizer_) {
    other.tokenizer_ = nullptr;
  }
  Tokenizer& operator=(Tokenizer&& other) {
    if (tokenizer_) iree_tokenizer_free(tokenizer_);
    tokenizer_ = other.tokenizer_;
    other.tokenizer_ = nullptr;
    return *this;
  }
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  ~Tokenizer() {
    if (tokenizer_) {
      iree_tokenizer_free(tokenizer_);
    }
  }

  static Tokenizer FromFile(std::string path) {
    // Memory-map the file (same approach as iree-tokenize CLI).
    iree_string_view_t path_view =
        iree_make_string_view(path.data(), path.size());
    iree_io_file_contents_t* contents = nullptr;
    CheckApiStatus(
        iree_io_file_contents_map(path_view, IREE_IO_FILE_ACCESS_READ,
                                  iree_allocator_system(), &contents),
        ("Failed to open file: " + path).c_str());
    iree_string_view_t file_data = {(const char*)contents->const_buffer.data,
                                    contents->const_buffer.data_length};

    // Format detection by file extension (same as iree-tokenize CLI).
    // .tiktoken → tiktoken format, everything else → HuggingFace JSON.
    // Parse directly from the mmap'd data — no copy to std::string.
    Tokenizer tok;
    iree_status_t status;
    if (iree_string_view_ends_with(path_view, IREE_SV(".tiktoken"))) {
      // Infer encoding name from filename: "cl100k_base.tiktoken" →
      // "cl100k_base"
      iree_host_size_t last_sep = iree_string_view_find_last_of(
          path_view, IREE_SV("/\\"), IREE_STRING_VIEW_NPOS);
      iree_string_view_t encoding =
          (last_sep != IREE_STRING_VIEW_NPOS)
              ? iree_string_view_substr(path_view, last_sep + 1,
                                        IREE_HOST_SIZE_MAX)
              : path_view;
      iree_string_view_consume_suffix(&encoding, IREE_SV(".tiktoken"));
      const iree_tokenizer_tiktoken_config_t* config =
          iree_tokenizer_tiktoken_config_by_name(encoding);
      if (!config) {
        iree_io_file_contents_free(contents);
        throw RaiseValueError(
            ("Unknown tiktoken encoding in filename: " + path +
             ". Expected one of: cl100k_base, o200k_base, o200k_harmony, "
             "r50k_base, gpt2, p50k_base, p50k_edit")
                .c_str());
      }
      {
        py::gil_scoped_release release;
        status = iree_tokenizer_from_tiktoken(
            file_data, config, iree_allocator_system(), &tok.tokenizer_);
      }
    } else {
      {
        py::gil_scoped_release release;
        status = iree_tokenizer_from_huggingface_json(
            file_data, iree_allocator_system(), &tok.tokenizer_);
      }
    }
    iree_io_file_contents_free(contents);
    CheckApiStatus(status, ("Failed to load tokenizer: " + path).c_str());
    return tok;
  }

  static Tokenizer FromHuggingfaceJson(std::string json) {
    Tokenizer tok;
    iree_string_view_t json_view = {json.data(), json.size()};
    iree_status_t status;
    {
      py::gil_scoped_release release;
      status = iree_tokenizer_from_huggingface_json(
          json_view, iree_allocator_system(), &tok.tokenizer_);
    }
    CheckApiStatus(status, "Failed to parse HuggingFace tokenizer JSON");
    return tok;
  }

  static Tokenizer FromTiktoken(std::string data, std::string encoding) {
    const iree_tokenizer_tiktoken_config_t* config =
        iree_tokenizer_tiktoken_config_by_name(
            iree_make_string_view(encoding.data(), encoding.size()));
    if (!config) {
      throw RaiseValueError(
          ("Unknown tiktoken encoding: " + encoding +
           ". Expected one of: cl100k_base, o200k_base, o200k_harmony, "
           "r50k_base, gpt2, p50k_base, p50k_edit")
              .c_str());
    }
    Tokenizer tok;
    iree_string_view_t data_view = {data.data(), data.size()};
    iree_status_t status;
    {
      py::gil_scoped_release release;
      status = iree_tokenizer_from_tiktoken(
          data_view, config, iree_allocator_system(), &tok.tokenizer_);
    }
    CheckApiStatus(status, "Failed to parse tiktoken data");
    return tok;
  }

  py::list Encode(std::string text, bool add_special_tokens,
                  bool no_special_token_matching) {
    iree_string_view_t text_view = {text.data(), text.size()};
    iree_tokenizer_encode_flags_t flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
    if (add_special_tokens) {
      flags |= IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS;
    }
    if (no_special_token_matching) {
      flags |= IREE_TOKENIZER_ENCODE_FLAG_NO_SPECIAL_TOKEN_MATCHING;
    }

    // At most 1 content token per byte, plus the post-processor's special
    // tokens (exact count from the C API). The retry loop handles the rare
    // case where this is still insufficient (RESOURCE_EXHAUSTED).
    iree_host_size_t special =
        iree_tokenizer_max_special_token_count(tokenizer_);
    iree_host_size_t capacity =
        std::max(text.size() + special, (size_t)kStreamTokenCapacity);
    std::vector<iree_tokenizer_token_id_t> token_ids(capacity);
    iree_host_size_t token_count = 0;

    while (true) {
      iree_tokenizer_token_output_t output = iree_tokenizer_make_token_output(
          token_ids.data(), NULL, NULL, capacity);
      iree_status_t status;
      {
        py::gil_scoped_release release;
        status = iree_tokenizer_encode(tokenizer_, text_view, flags, output,
                                       iree_allocator_system(), &token_count);
      }
      if (iree_status_is_ok(status)) break;
      if (!iree_status_is_resource_exhausted(status)) {
        CheckApiStatus(status, "Tokenizer encode failed");
      }
      iree_status_ignore(status);
      capacity *= 2;
      token_ids.resize(capacity);
    }

    py::list result;
    for (iree_host_size_t i = 0; i < token_count; ++i) {
      result.append(token_ids[i]);
    }
    return result;
  }

  py::str Decode(std::vector<int32_t> token_ids, bool skip_special_tokens) {
    iree_tokenizer_token_id_list_t tokens = {
        token_ids.size(),
        token_ids.data(),
    };
    iree_tokenizer_decode_flags_t flags = IREE_TOKENIZER_DECODE_FLAG_NONE;
    if (skip_special_tokens) {
      flags |= IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS;
    }

    iree_host_size_t capacity =
        std::max(token_ids.size() * 8, (size_t)kDecodeBufferSize);
    std::vector<char> text_buf(capacity);
    iree_host_size_t text_length = 0;

    while (true) {
      iree_mutable_string_view_t text_output = {text_buf.data(), capacity};
      iree_status_t status;
      {
        py::gil_scoped_release release;
        status = iree_tokenizer_decode(tokenizer_, tokens, flags, text_output,
                                       iree_allocator_system(), &text_length);
      }
      if (iree_status_is_ok(status)) break;
      if (!iree_status_is_resource_exhausted(status)) {
        CheckApiStatus(status, "Tokenizer decode failed");
      }
      iree_status_ignore(status);
      capacity *= 2;
      text_buf.resize(capacity);
    }

    return py::str(text_buf.data(), text_length);
  }

  iree_host_size_t vocab_size() const {
    return iree_tokenizer_vocab_token_count(iree_tokenizer_vocab(tokenizer_));
  }

  std::string model_type() const {
    iree_string_view_t name = iree_tokenizer_model_type_name(tokenizer_);
    return std::string(name.data, name.size);
  }

  py::object id_to_token(int32_t id) const {
    if (id < 0) return py::none();
    const iree_tokenizer_vocab_t* vocab = iree_tokenizer_vocab(tokenizer_);
    if ((iree_host_size_t)id >= iree_tokenizer_vocab_capacity(vocab)) {
      return py::none();
    }
    // Check for unused gap slots in sparse vocabs (marked ATTR_UNUSED by the
    // builder). Empty-string tokens are valid and must NOT be treated as gaps.
    iree_tokenizer_token_attr_t attrs =
        iree_tokenizer_vocab_token_attrs(vocab, id);
    if (iree_any_bit_set(attrs, IREE_TOKENIZER_TOKEN_ATTR_UNUSED)) {
      return py::none();
    }
    iree_string_view_t text = iree_tokenizer_vocab_token_text(vocab, id);
    return py::str(text.data, text.size);
  }

  py::object token_to_id(std::string token) const {
    const iree_tokenizer_vocab_t* vocab = iree_tokenizer_vocab(tokenizer_);
    iree_string_view_t text_view = {token.data(), token.size()};
    int32_t id = iree_tokenizer_vocab_lookup(vocab, text_view);
    if (id < 0) return py::none();
    return py::int_(id);
  }

  py::dict special_ids() const {
    const iree_tokenizer_vocab_t* vocab = iree_tokenizer_vocab(tokenizer_);
    iree_tokenizer_special_ids_t ids = iree_tokenizer_vocab_special_ids(vocab);
    auto to_py = [](int32_t id) -> py::object {
      return id >= 0 ? py::int_(id) : py::none();
    };
    py::dict result;
    result["bos"] = to_py(ids.bos);
    result["eos"] = to_py(ids.eos);
    result["unk"] = to_py(ids.unk);
    result["pad"] = to_py(ids.pad);
    result["sep"] = to_py(ids.sep);
    result["cls"] = to_py(ids.cls);
    result["mask"] = to_py(ids.mask);
    return result;
  }

 private:
  friend class EncodeStream;
  friend class DecodeStream;
  iree_tokenizer_t* tokenizer_;
};

// ============================================================================
// EncodeStream
// ============================================================================

class EncodeStream {
 public:
  EncodeStream(Tokenizer& tokenizer, bool add_special_tokens,
               bool no_special_token_matching)
      : state_(nullptr), finalized_(false) {
    iree_tokenizer_t* raw = tokenizer.tokenizer_;

    iree_tokenizer_encode_flags_t flags =
        IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START;
    if (add_special_tokens) {
      flags |= IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS;
    }
    if (no_special_token_matching) {
      flags |= IREE_TOKENIZER_ENCODE_FLAG_NO_SPECIAL_TOKEN_MATCHING;
    }

    // Allocate state storage.
    iree_host_size_t state_size = 0;
    CheckApiStatus(iree_tokenizer_encode_state_calculate_size(raw, &state_size),
                   "Failed to calculate encode state size");
    state_storage_.resize(state_size);

    // Allocate transform buffer.
    iree_host_size_t tb_size =
        iree_tokenizer_transform_buffer_recommended_size(kStreamChunkSizeHint);
    transform_buffer_.resize(tb_size);

    iree_byte_span_t state_span =
        iree_make_byte_span(state_storage_.data(), state_storage_.size());
    iree_byte_span_t tb_span =
        iree_make_byte_span(transform_buffer_.data(), transform_buffer_.size());

    CheckApiStatus(iree_tokenizer_encode_state_initialize(
                       raw, state_span, tb_span,
                       iree_tokenizer_offset_run_list_empty(), flags, &state_),
                   "Failed to initialize encode state");
  }

  ~EncodeStream() {
    if (state_) {
      iree_tokenizer_encode_state_deinitialize(state_);
    }
  }

  EncodeStream(const EncodeStream&) = delete;
  EncodeStream& operator=(const EncodeStream&) = delete;

  py::list Feed(std::string chunk) {
    py::ft_lock_guard lock(mutex_);
    if (finalized_) {
      throw RaiseValueError("Cannot feed after finalize");
    }

    iree_string_view_t chunk_view = {chunk.data(), chunk.size()};
    py::list result;

    while (chunk_view.size > 0) {
      iree_tokenizer_token_output_t output = iree_tokenizer_make_token_output(
          token_ids_.data(), NULL, NULL, kStreamTokenCapacity);
      iree_host_size_t bytes_consumed = 0;
      iree_host_size_t token_count = 0;

      iree_status_t status;
      {
        py::gil_scoped_release release;
        status = iree_tokenizer_encode_state_feed(
            state_, chunk_view, output, &bytes_consumed, &token_count);
      }
      CheckApiStatus(status, "Encode stream feed failed");

      for (iree_host_size_t i = 0; i < token_count; ++i) {
        result.append(token_ids_[i]);
      }

      // Zero-progress guard: the C API should always make progress, but
      // defend against infinite loops if it doesn't (same pattern as
      // DecodeStream::FeedTokens).
      if (bytes_consumed == 0 && token_count == 0) break;

      chunk_view.data += bytes_consumed;
      chunk_view.size -= bytes_consumed;
    }

    return result;
  }

  py::list Finalize() {
    py::ft_lock_guard lock(mutex_);
    if (finalized_) {
      throw RaiseValueError("Already finalized");
    }

    // The C finalize is non-retryable: it consumes pipeline state
    // destructively, so the output buffer must be large enough on the first
    // call. Query pending_token_bound for a tight upper bound derived from
    // the actual pipeline state (ring buffer, model, normalizer, post-
    // processor). This is always >= the actual token count from finalize.
    iree_host_size_t capacity =
        std::max(iree_tokenizer_encode_state_pending_token_bound(state_),
                 (size_t)kStreamTokenCapacity);
    if (capacity > token_ids_.size()) {
      token_ids_.resize(capacity);
    }

    iree_tokenizer_token_output_t output = iree_tokenizer_make_token_output(
        token_ids_.data(), NULL, NULL, capacity);
    iree_host_size_t token_count = 0;

    // Mark finalized before the C call. The C state enters finalize mode
    // unconditionally and is not safely re-finalizable after a failure.
    finalized_ = true;
    iree_status_t status;
    {
      py::gil_scoped_release release;
      status =
          iree_tokenizer_encode_state_finalize(state_, output, &token_count);
    }
    CheckApiStatus(status, "Encode stream finalize failed");

    py::list result;
    for (iree_host_size_t i = 0; i < token_count; ++i) {
      result.append(token_ids_[i]);
    }
    return result;
  }

 private:
  iree_tokenizer_encode_state_t* state_;
  std::vector<uint8_t> state_storage_;
  std::vector<uint8_t> transform_buffer_;
  std::vector<iree_tokenizer_token_id_t> token_ids_ =
      std::vector<iree_tokenizer_token_id_t>(kStreamTokenCapacity);
  bool finalized_;
  py::ft_mutex mutex_;  // Serializes access in free-threaded Python.

 public:
  bool is_finalized() const { return finalized_; }
};

// ============================================================================
// DecodeStream
// ============================================================================

class DecodeStream {
 public:
  DecodeStream(Tokenizer& tokenizer, bool skip_special_tokens)
      : state_(nullptr), finalized_(false) {
    iree_tokenizer_t* raw = tokenizer.tokenizer_;

    iree_tokenizer_decode_flags_t flags = IREE_TOKENIZER_DECODE_FLAG_NONE;
    if (skip_special_tokens) {
      flags |= IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS;
    }

    iree_host_size_t state_size = 0;
    CheckApiStatus(iree_tokenizer_decode_state_calculate_size(raw, &state_size),
                   "Failed to calculate decode state size");
    state_storage_.resize(state_size);

    iree_byte_span_t state_span =
        iree_make_byte_span(state_storage_.data(), state_storage_.size());

    CheckApiStatus(
        iree_tokenizer_decode_state_initialize(raw, flags, state_span, &state_),
        "Failed to initialize decode state");
  }

  ~DecodeStream() {
    if (state_) {
      iree_tokenizer_decode_state_deinitialize(state_);
    }
  }

  DecodeStream(const DecodeStream&) = delete;
  DecodeStream& operator=(const DecodeStream&) = delete;

  py::str Feed(std::vector<int32_t> token_ids) {
    py::ft_lock_guard lock(mutex_);
    if (finalized_) {
      throw RaiseValueError("Cannot feed after finalize");
    }
    iree_tokenizer_token_id_list_t tokens = {
        token_ids.size(),
        token_ids.data(),
    };
    return FeedTokens(tokens);
  }

  // Fast path for single-token feed (avoids list->vector conversion).
  py::str FeedOne(int32_t token_id) {
    py::ft_lock_guard lock(mutex_);
    if (finalized_) {
      throw RaiseValueError("Cannot feed after finalize");
    }
    iree_tokenizer_token_id_list_t tokens = {1, &token_id};
    return FeedTokens(tokens);
  }

  py::str Finalize() {
    py::ft_lock_guard lock(mutex_);
    if (finalized_) {
      throw RaiseValueError("Already finalized");
    }

    // Mark finalized before the C call. The C state is not safely
    // re-finalizable after entering finalize mode.
    finalized_ = true;

    // The decoder finalize flushes pending byte-fallback data. Loop until
    // it produces no more output. Double the buffer on RESOURCE_EXHAUSTED
    // (same pattern as FeedTokens).
    std::string result;
    iree_host_size_t text_length = 0;
    do {
      iree_mutable_string_view_t text_output = {text_buf_.data(),
                                                text_buf_.size()};
      text_length = 0;

      iree_status_t status;
      {
        py::gil_scoped_release release;
        status = iree_tokenizer_decode_state_finalize(state_, text_output,
                                                      &text_length);
      }
      if (iree_status_is_resource_exhausted(status)) {
        iree_status_ignore(status);
        text_buf_.resize(text_buf_.size() * 2);
        continue;
      }
      CheckApiStatus(status, "Decode stream finalize failed");

      if (text_length > 0) {
        result.append(text_buf_.data(), text_length);
      }
    } while (text_length > 0);

    return py::str(result.data(), result.size());
  }

 private:
  // Shared implementation for Feed() and FeedOne(). Loops until all tokens
  // are consumed, handling byte-fallback decoders that can flush pending text
  // without consuming the current token, and zero-progress buffer doubling.
  py::str FeedTokens(iree_tokenizer_token_id_list_t tokens) {
    std::string result;
    while (tokens.count > 0) {
      iree_mutable_string_view_t text_output = {text_buf_.data(),
                                                text_buf_.size()};
      iree_host_size_t tokens_consumed = 0;
      iree_host_size_t text_length = 0;

      iree_status_t status;
      {
        py::gil_scoped_release release;
        status = iree_tokenizer_decode_state_feed(
            state_, tokens, text_output, &tokens_consumed, &text_length);
      }
      CheckApiStatus(status, "Decode stream feed failed");

      if (text_length > 0) {
        result.append(text_buf_.data(), text_length);
      }

      // Zero-progress guard: output buffer too small for next token.
      if (tokens_consumed == 0 && text_length == 0) {
        text_buf_.resize(text_buf_.size() * 2);
        continue;
      }

      tokens.values += tokens_consumed;
      tokens.count -= tokens_consumed;
    }
    return py::str(result.data(), result.size());
  }

  iree_tokenizer_decode_state_t* state_;
  std::vector<uint8_t> state_storage_;
  // Reused across Feed()/Finalize() calls to avoid per-call heap allocation.
  std::vector<char> text_buf_ = std::vector<char>(kDecodeBufferSize);
  bool finalized_;
  py::ft_mutex mutex_;  // Serializes access in free-threaded Python.

 public:
  bool is_finalized() const { return finalized_; }
};

// ============================================================================
// Bindings
// ============================================================================

void SetupTokenizerBindings(py::module_& m) {
  auto encode_stream =
      py::class_<EncodeStream>(m, "TokenizerEncodeStream")
          .def("__repr__",
               [](EncodeStream&) { return "<TokenizerEncodeStream>"; })
          .def("feed", &EncodeStream::Feed, py::arg("chunk"),
               "Feed a text chunk. Returns tokens produced from this chunk.")
          .def("finalize", &EncodeStream::Finalize,
               "Flush remaining tokens. Must be called after all input is fed.")
          .def("__enter__", [](py::object self) -> py::object { return self; })
          .def("__exit__", [](EncodeStream& self, py::args) {
            // Auto-finalize if not already done, discarding the result.
            // This ensures the C state is properly finalized even if the
            // user forgets to call finalize() explicitly.
            if (!self.is_finalized()) {
              try {
                self.Finalize();
              } catch (...) {
                // Suppress errors during cleanup (same as file.close()).
              }
            }
          });

  auto decode_stream =
      py::class_<DecodeStream>(m, "TokenizerDecodeStream")
          .def("__repr__",
               [](DecodeStream&) { return "<TokenizerDecodeStream>"; })
          .def("feed", &DecodeStream::Feed, py::arg("token_ids"),
               "Feed token IDs. Returns text produced from these tokens.")
          .def("feed_one", &DecodeStream::FeedOne, py::arg("token_id"),
               "Feed a single token ID. Faster than feed([id]) for per-token "
               "LLM decoding.")
          .def("finalize", &DecodeStream::Finalize,
               "Flush remaining text. Must be called after all tokens are fed.")
          .def("__enter__", [](py::object self) -> py::object { return self; })
          .def("__exit__", [](DecodeStream& self, py::args) {
            if (!self.is_finalized()) {
              try {
                self.Finalize();
              } catch (...) {
              }
            }
          });

  py::class_<Tokenizer>(m, "Tokenizer")
      .def("__repr__",
           [](Tokenizer& self) {
             return "<Tokenizer model_type='" + self.model_type() +
                    "' vocab_size=" + std::to_string(self.vocab_size()) + ">";
           })
      .def_static(
          "from_file",
          [](py::object path) {
            // Use os.fspath() to accept str, bytes, and os.PathLike objects,
            // and reject nonsensical types like int with a clean TypeError.
            py::object os = py::module_::import_("os");
            py::object fs_path = os.attr("fspath")(path);
            // os.fspath() may return str or bytes. Decode bytes to str.
            std::string path_str;
            if (py::isinstance<py::bytes>(fs_path)) {
              path_str = py::cast<std::string>(
                  fs_path.attr("decode")(py::str("utf-8")));
            } else {
              path_str = py::cast<std::string>(fs_path);
            }
            return Tokenizer::FromFile(std::move(path_str));
          },
          py::arg("path"),
          "Load a tokenizer from a file path (str or os.PathLike). "
          "Auto-detects HuggingFace JSON and tiktoken formats.")
      .def_static("from_huggingface_json", &Tokenizer::FromHuggingfaceJson,
                  py::arg("json"),
                  "Create a tokenizer from a HuggingFace tokenizer.json "
                  "string.")
      .def_static("from_tiktoken", &Tokenizer::FromTiktoken, py::arg("data"),
                  py::arg("encoding"),
                  "Create a tokenizer from tiktoken data and encoding name. "
                  "Supported encodings: cl100k_base, o200k_base, "
                  "o200k_harmony, r50k_base, gpt2, p50k_base, p50k_edit.")
      .def("encode", &Tokenizer::Encode, py::arg("text"),
           py::arg("add_special_tokens") = false,
           py::arg("no_special_token_matching") = false,
           "Encode text to token IDs. Set no_special_token_matching=True "
           "to treat special tokens as literal text (like tiktoken's "
           "encode_ordinary).")
      .def("decode", &Tokenizer::Decode, py::arg("token_ids"),
           py::arg("skip_special_tokens") = false, "Decode token IDs to text.")
      .def(
          "encode_stream",
          [](Tokenizer& self, bool add_special_tokens,
             bool no_special_token_matching) {
            return new EncodeStream(self, add_special_tokens,
                                    no_special_token_matching);
          },
          py::arg("add_special_tokens") = false,
          py::arg("no_special_token_matching") = false, py::keep_alive<0, 1>(),
          "Create a streaming encoder.")
      .def(
          "decode_stream",
          [](Tokenizer& self, bool skip_special_tokens) {
            return new DecodeStream(self, skip_special_tokens);
          },
          py::arg("skip_special_tokens") = false, py::keep_alive<0, 1>(),
          "Create a streaming decoder.")
      .def_prop_ro("vocab_size", &Tokenizer::vocab_size,
                   "Number of active tokens in the vocabulary.")
      .def_prop_ro("model_type", &Tokenizer::model_type,
                   "Model type name (e.g., 'BPE', 'WordPiece', 'Unigram').")
      .def("id_to_token", &Tokenizer::id_to_token, py::arg("token_id"),
           "Get the text for a token ID. Returns None if out of range.")
      .def("token_to_id", &Tokenizer::token_to_id, py::arg("token"),
           "Look up a token string. Returns None if not found.")
      .def_prop_ro("special_ids", &Tokenizer::special_ids,
                   "Dict of special token IDs (bos, eos, unk, pad, sep, cls, "
                   "mask). Value is None if absent.");
}

}  // namespace iree::python
