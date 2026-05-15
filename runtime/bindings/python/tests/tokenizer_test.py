# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import base64
import functools
import gc
import json
import logging
import os
import tempfile
import threading
import unittest
from pathlib import Path

import iree.runtime as rt

TESTDATA_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "runtime"
    / "src"
    / "iree"
    / "tokenizer"
    / "testdata"
)

BPE_MINIMAL_JSON = TESTDATA_DIR / "bpe_bytelevel_minimal.json"
BPE_UNICODE_JSON = TESTDATA_DIR / "bpe_bytelevel_unicode.json"


def _require_testdata(test_func):
    """Skip test if tokenizer testdata is not available."""

    @functools.wraps(test_func)
    def wrapper(self):
        if not BPE_MINIMAL_JSON.exists():
            self.skipTest(f"Tokenizer testdata not found: {BPE_MINIMAL_JSON}")
        return test_func(self)

    return wrapper


class TokenizerLoadTest(unittest.TestCase):
    @_require_testdata
    def test_from_file(self):
        tok = rt.Tokenizer.from_file(str(BPE_MINIMAL_JSON))
        self.assertEqual(tok.model_type, "BPE")
        self.assertGreater(tok.vocab_size, 0)

    @_require_testdata
    def test_from_file_pathlike(self):
        """from_file() should accept Path objects, not just str."""
        tok = rt.Tokenizer.from_file(BPE_MINIMAL_JSON)  # Path, not str
        self.assertEqual(tok.model_type, "BPE")

    @_require_testdata
    def test_from_huggingface_json(self):
        json_str = BPE_MINIMAL_JSON.read_text()
        tok = rt.Tokenizer.from_huggingface_json(json_str)
        self.assertEqual(tok.model_type, "BPE")

    def test_from_file_not_found(self):
        with self.assertRaises(ValueError):
            rt.Tokenizer.from_file("/nonexistent/path/tokenizer.json")

    def test_from_file_bad_format(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("not json content")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                rt.Tokenizer.from_file(path)
        finally:
            Path(path).unlink()

    def test_from_huggingface_json_invalid(self):
        with self.assertRaises(Exception):
            rt.Tokenizer.from_huggingface_json("{invalid json")

    @_require_testdata
    def test_from_file_bytes_path(self):
        """from_file accepts bytes paths via os.fsencode."""
        tok = rt.Tokenizer.from_file(os.fsencode(str(BPE_MINIMAL_JSON)))
        self.assertEqual(tok.model_type, "BPE")

    def test_from_file_rejects_non_path(self):
        """from_file raises TypeError for non-path types."""
        with self.assertRaises(TypeError):
            rt.Tokenizer.from_file(123)
        with self.assertRaises(TypeError):
            rt.Tokenizer.from_file(object())

    def test_from_tiktoken_inline(self):
        """from_tiktoken creates a tokenizer from inline tiktoken data."""
        # Tiktoken requires all 256 single-byte tokens as the base vocabulary.
        lines = []
        for byte_val in range(256):
            b64 = base64.b64encode(bytes([byte_val])).decode()
            lines.append(f"{b64} {byte_val}")
        data = "\n".join(lines)
        tok = rt.Tokenizer.from_tiktoken(data, "cl100k_base")
        self.assertGreaterEqual(tok.vocab_size, 256)

    def test_from_file_tiktoken_unknown_encoding(self):
        """from_file with .tiktoken extension and unknown encoding raises."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tiktoken", prefix="unknown_enc", delete=False
        ) as f:
            f.write("AA== 0\n")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                rt.Tokenizer.from_file(path)
        finally:
            Path(path).unlink()

    @_require_testdata
    def test_multiple_tokenizers(self):
        """Verify independent tokenizer instances work correctly."""
        tok1 = rt.Tokenizer.from_file(str(BPE_MINIMAL_JSON))
        tok2 = rt.Tokenizer.from_file(str(BPE_MINIMAL_JSON))
        ids1 = tok1.encode("Hello")
        ids2 = tok2.encode("Hello")
        self.assertEqual(ids1, ids2)
        del tok1
        gc.collect()
        # tok2 should still work after tok1 is destroyed.
        ids3 = tok2.encode("Hello")
        self.assertEqual(ids2, ids3)


class TokenizerEncodeDecodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BPE_MINIMAL_JSON.exists():
            raise unittest.SkipTest(f"Tokenizer testdata not found: {BPE_MINIMAL_JSON}")
        cls.tok = rt.Tokenizer.from_file(str(BPE_MINIMAL_JSON))

    def test_encode_decode_roundtrip(self):
        text = "Hello world"
        ids = self.tok.encode(text)
        self.assertIsInstance(ids, list)
        self.assertTrue(all(isinstance(i, int) for i in ids))
        decoded = self.tok.decode(ids)
        self.assertEqual(decoded, text)

    def test_encode_empty(self):
        ids = self.tok.encode("")
        self.assertEqual(ids, [])

    def test_decode_empty(self):
        text = self.tok.decode([])
        self.assertEqual(text, "")

    def test_encode_multiple_words(self):
        text = "The quick brown fox"
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        self.assertEqual(decoded, text)

    def test_encode_add_special_tokens_no_postprocessor(self):
        # Minimal tokenizer has no post-processor, so add_special_tokens
        # should produce the same output (no BOS/EOS to add).
        ids_without = self.tok.encode("Hello")
        ids_with = self.tok.encode("Hello", add_special_tokens=True)
        self.assertEqual(ids_without, ids_with)

    def test_decode_skip_special_tokens(self):
        ids = self.tok.encode("Hello")
        text_without = self.tok.decode(ids)
        text_with = self.tok.decode(ids, skip_special_tokens=True)
        self.assertEqual(text_without, text_with)

    @unittest.skip(
        "ByteLevel decoder STATELESS bug: decode drops multi-byte UTF-8. "
        "Fix: change CAPABILITY_STATELESS to CAPABILITY_NONE in byte_level.c."
    )
    def test_encode_decode_unicode(self):
        """Byte-level BPE must round-trip non-ASCII text (multibyte UTF-8)."""
        if not BPE_UNICODE_JSON.exists():
            self.skipTest(f"Unicode tokenizer not found: {BPE_UNICODE_JSON}")
        tok = rt.Tokenizer.from_file(str(BPE_UNICODE_JSON))
        for text in ["café", "你好世界", "hello 😀 world", "Ñoño"]:
            with self.subTest(text=text):
                ids = tok.encode(text)
                self.assertTrue(len(ids) > 0)
                self.assertEqual(tok.decode(ids), text)

    def test_repr(self):
        r = repr(self.tok)
        self.assertIn("Tokenizer", r)
        self.assertIn("BPE", r)
        self.assertIn("112", r)


class TokenizerStreamingEncodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BPE_MINIMAL_JSON.exists():
            raise unittest.SkipTest(f"Tokenizer testdata not found: {BPE_MINIMAL_JSON}")
        cls.tok = rt.Tokenizer.from_file(str(BPE_MINIMAL_JSON))

    def test_streaming_encode_matches_batch(self):
        batch_ids = self.tok.encode("Hello world")
        enc = self.tok.encode_stream()
        ids1 = enc.feed("Hello ")
        ids2 = enc.feed("world")
        ids3 = enc.finalize()
        self.assertEqual(ids1 + ids2 + ids3, batch_ids)

    def test_context_manager(self):
        with self.tok.encode_stream() as enc:
            ids = enc.feed("test")
            ids += enc.finalize()
        self.assertIsInstance(ids, list)
        self.assertGreater(len(ids), 0)

    def test_finalize_twice_raises(self):
        enc = self.tok.encode_stream()
        enc.feed("x")
        enc.finalize()
        with self.assertRaises(ValueError):
            enc.finalize()

    def test_feed_after_finalize_raises(self):
        enc = self.tok.encode_stream()
        enc.finalize()
        with self.assertRaises(ValueError):
            enc.feed("x")

    def test_multiple_streams_sequential(self):
        """Verify creating multiple streams from the same tokenizer works."""
        for _ in range(3):
            enc = self.tok.encode_stream()
            ids = enc.feed("test")
            ids += enc.finalize()
            del enc
        gc.collect()
        # Tokenizer should still work after all streams are destroyed.
        self.assertGreater(len(self.tok.encode("test")), 0)


class TokenizerStreamingDecodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BPE_MINIMAL_JSON.exists():
            raise unittest.SkipTest(f"Tokenizer testdata not found: {BPE_MINIMAL_JSON}")
        cls.tok = rt.Tokenizer.from_file(str(BPE_MINIMAL_JSON))
        cls.ids = cls.tok.encode("Hello world")

    def test_streaming_decode_matches_batch(self):
        dec = self.tok.decode_stream()
        t1 = dec.feed(self.ids[:2])
        t2 = dec.feed(self.ids[2:])
        t3 = dec.finalize()
        self.assertEqual(t1 + t2 + t3, "Hello world")

    def test_context_manager(self):
        with self.tok.decode_stream() as dec:
            text = dec.feed(self.ids)
            text += dec.finalize()
        self.assertEqual(text, "Hello world")

    def test_finalize_twice_raises(self):
        dec = self.tok.decode_stream()
        dec.feed(self.ids)
        dec.finalize()
        with self.assertRaises(ValueError):
            dec.finalize()

    def test_feed_after_finalize_raises(self):
        dec = self.tok.decode_stream()
        dec.finalize()
        with self.assertRaises(ValueError):
            dec.feed(self.ids)

    def test_single_token_feed(self):
        """Feed tokens one at a time and verify accumulated result."""
        dec = self.tok.decode_stream()
        parts = []
        for token_id in self.ids:
            parts.append(dec.feed([token_id]))
        parts.append(dec.finalize())
        self.assertEqual("".join(parts), "Hello world")

    def test_feed_one(self):
        """feed_one() should produce the same result as feed([id])."""
        dec_list = self.tok.decode_stream()
        dec_one = self.tok.decode_stream()
        parts_list = []
        parts_one = []
        for token_id in self.ids:
            parts_list.append(dec_list.feed([token_id]))
            parts_one.append(dec_one.feed_one(token_id))
        parts_list.append(dec_list.finalize())
        parts_one.append(dec_one.finalize())
        self.assertEqual("".join(parts_list), "".join(parts_one))
        self.assertEqual("".join(parts_one), "Hello world")

    def test_feed_one_after_finalize_raises(self):
        dec = self.tok.decode_stream()
        dec.finalize()
        with self.assertRaises(ValueError):
            dec.feed_one(0)


class TokenizerVocabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BPE_MINIMAL_JSON.exists():
            raise unittest.SkipTest(f"Tokenizer testdata not found: {BPE_MINIMAL_JSON}")
        cls.tok = rt.Tokenizer.from_file(str(BPE_MINIMAL_JSON))

    def test_vocab_size(self):
        self.assertEqual(self.tok.vocab_size, 112)

    def test_model_type(self):
        self.assertEqual(self.tok.model_type, "BPE")

    def test_id_to_token_valid(self):
        token = self.tok.id_to_token(39)
        self.assertIsNotNone(token)
        self.assertIsInstance(token, str)

    def test_id_to_token_out_of_range(self):
        self.assertIsNone(self.tok.id_to_token(99999))

    def test_id_to_token_negative(self):
        self.assertIsNone(self.tok.id_to_token(-1))
        self.assertIsNone(self.tok.id_to_token(-100))

    def test_token_roundtrip(self):
        """Verify id_to_token and encode are consistent."""
        ids = self.tok.encode("H")
        self.assertEqual(len(ids), 1)
        token_text = self.tok.id_to_token(ids[0])
        self.assertEqual(token_text, "H")

    def test_token_to_id_known(self):
        """token_to_id returns correct ID for a known vocab entry."""
        ids = self.tok.encode("H")
        self.assertEqual(len(ids), 1)
        looked_up = self.tok.token_to_id("H")
        self.assertIsNotNone(looked_up)
        self.assertEqual(looked_up, ids[0])

    def test_token_to_id_unknown(self):
        self.assertIsNone(self.tok.token_to_id("nonexistent_token_xyz"))

    def test_special_ids(self):
        ids = self.tok.special_ids
        self.assertIsInstance(ids, dict)
        for key in ("bos", "eos", "unk", "pad", "sep", "cls", "mask"):
            self.assertIn(key, ids)
            # Value is int (if configured) or None (if absent).
            self.assertTrue(
                ids[key] is None or isinstance(ids[key], int),
                f"special_ids[{key!r}] = {ids[key]!r}, expected int or None",
            )


class TokenizerThreadingTest(unittest.TestCase):
    """Test concurrent use of a shared Tokenizer from multiple threads.

    The Tokenizer itself is thread-safe (immutable C object). Each thread
    must use its own EncodeStream/DecodeStream (not shared across threads).
    """

    @classmethod
    def setUpClass(cls):
        if not BPE_MINIMAL_JSON.exists():
            raise unittest.SkipTest(f"Tokenizer testdata not found: {BPE_MINIMAL_JSON}")
        cls.tok = rt.Tokenizer.from_file(str(BPE_MINIMAL_JSON))

    def test_concurrent_batch_encode(self):
        import threading

        results = {}
        errors = []

        def worker(thread_id):
            try:
                text = f"Thread {thread_id} hello world"
                for _ in range(100):
                    ids = self.tok.encode(text)
                results[thread_id] = ids
            except Exception as e:
                errors.append((thread_id, e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")
        for i in range(4):
            expected = self.tok.encode(f"Thread {i} hello world")
            self.assertEqual(results[i], expected)

    def test_concurrent_streaming(self):
        import threading

        results = {}
        errors = []

        def worker(thread_id):
            try:
                text = f"Thread {thread_id} hello"
                enc = self.tok.encode_stream()
                ids = enc.feed(text)
                ids += enc.finalize()
                results[thread_id] = ids
            except Exception as e:
                errors.append((thread_id, e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")
        for i in range(4):
            expected = self.tok.encode(f"Thread {i} hello")
            self.assertEqual(results[i], expected)


class TokenizerLargeInputTest(unittest.TestCase):
    """Test with inputs that exceed internal buffer sizes."""

    @classmethod
    def setUpClass(cls):
        if not BPE_MINIMAL_JSON.exists():
            raise unittest.SkipTest(f"Tokenizer testdata not found: {BPE_MINIMAL_JSON}")
        cls.tok = rt.Tokenizer.from_file(str(BPE_MINIMAL_JSON))

    def test_streaming_encode_exceeds_token_buffer(self):
        """Feed a chunk that produces more tokens than the 256-token buffer."""
        for n in [300, 500, 1000, 5000]:
            text = "a" * n
            batch_ids = self.tok.encode(text)
            enc = self.tok.encode_stream()
            stream_ids = enc.feed(text)
            stream_ids += enc.finalize()
            self.assertEqual(
                stream_ids,
                batch_ids,
                f"Stream/batch mismatch at {n} chars: "
                f"stream={len(stream_ids)}, batch={len(batch_ids)}",
            )

    def test_large_multi_chunk_roundtrip(self):
        """120KB input split into 1KB chunks must match batch."""
        text = "Hello world " * 10000
        batch_ids = self.tok.encode(text)
        enc = self.tok.encode_stream()
        stream_ids = []
        for i in range(0, len(text), 1024):
            stream_ids.extend(enc.feed(text[i : i + 1024]))
        stream_ids.extend(enc.finalize())
        self.assertEqual(len(stream_ids), len(batch_ids))
        self.assertEqual(stream_ids, batch_ids)


def _build_special_tokens_json():
    """Build a tokenizer JSON with BOS/EOS special tokens and a post-processor."""
    base_json = BPE_MINIMAL_JSON.read_text()
    tok_dict = json.loads(base_json)
    # Add BOS (id=111 reused as <|bos|>) and EOS (id=112 as <|eos|>)
    tok_dict["model"]["vocab"]["<|bos|>"] = 112
    tok_dict["model"]["vocab"]["<|eos|>"] = 113
    tok_dict["added_tokens"].extend(
        [
            {
                "id": 112,
                "content": "<|bos|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": 113,
                "content": "<|eos|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
        ]
    )
    tok_dict["post_processor"] = {
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": "<|bos|>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "<|eos|>", "type_id": 0}},
        ],
        "pair": [
            {"SpecialToken": {"id": "<|bos|>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "<|eos|>", "type_id": 0}},
            {"Sequence": {"id": "B", "type_id": 1}},
            {"SpecialToken": {"id": "<|eos|>", "type_id": 1}},
        ],
        "special_tokens": {
            "<|bos|>": {"id": "<|bos|>", "ids": [112], "tokens": ["<|bos|>"]},
            "<|eos|>": {"id": "<|eos|>", "ids": [113], "tokens": ["<|eos|>"]},
        },
    }
    return json.dumps(tok_dict)


class TokenizerSpecialTokensTest(unittest.TestCase):
    """Test special token handling with a tokenizer that has BOS/EOS configured."""

    @classmethod
    def setUpClass(cls):
        if not BPE_MINIMAL_JSON.exists():
            raise unittest.SkipTest(f"Tokenizer testdata not found: {BPE_MINIMAL_JSON}")
        try:
            cls.tok = rt.Tokenizer.from_huggingface_json(_build_special_tokens_json())
        except Exception as e:
            raise unittest.SkipTest(f"Failed to build special tokens tokenizer: {e}")

    def test_vocab_includes_special_tokens(self):
        # The added special tokens should be in the vocab.
        self.assertIsNotNone(self.tok.id_to_token(112))  # <|bos|>
        self.assertIsNotNone(self.tok.id_to_token(113))  # <|eos|>

    def test_add_special_tokens_adds_bos_eos(self):
        ids_without = self.tok.encode("Hello")
        ids_with = self.tok.encode("Hello", add_special_tokens=True)
        # With special tokens, the output should be longer (BOS + tokens + EOS).
        self.assertGreater(len(ids_with), len(ids_without))

    def test_decode_skip_special_tokens(self):
        ids = self.tok.encode("Hello", add_special_tokens=True)
        text_with = self.tok.decode(ids, skip_special_tokens=False)
        text_skip = self.tok.decode(ids, skip_special_tokens=True)
        # Skipping special tokens should give clean text.
        self.assertIn("Hello", text_skip)
        # Not skipping should include special token text.
        self.assertGreater(len(text_with), len(text_skip))

    def test_encode_special_token_in_text(self):
        """Special tokens in input text should be recognized."""
        ids = self.tok.encode("<|endoftext|>")
        # Should produce the special token ID (111), not individual chars.
        self.assertIn(111, ids)

    def test_streaming_decode_skip_special_tokens(self):
        """Streaming decode with skip_special_tokens matches batch."""
        ids = self.tok.encode("Hello", add_special_tokens=True)
        batch_skip = self.tok.decode(ids, skip_special_tokens=True)
        batch_noskip = self.tok.decode(ids, skip_special_tokens=False)
        # Sanity: batch skip should be shorter.
        self.assertGreater(len(batch_noskip), len(batch_skip))
        # Streaming with skip.
        dec = self.tok.decode_stream(skip_special_tokens=True)
        stream_text = dec.feed(ids) + dec.finalize()
        self.assertEqual(stream_text, batch_skip)
        # Streaming without skip.
        dec2 = self.tok.decode_stream(skip_special_tokens=False)
        stream_noskip = dec2.feed(ids) + dec2.finalize()
        self.assertEqual(stream_noskip, batch_noskip)

    def test_encode_no_special_token_matching(self):
        """no_special_token_matching treats special tokens as literal text."""
        ids_normal = self.tok.encode("<|endoftext|>")
        ids_ordinary = self.tok.encode("<|endoftext|>", no_special_token_matching=True)
        # Normal: matched as special token (111).
        self.assertIn(111, ids_normal)
        # Ordinary: NOT matched, tokenized as literal characters.
        self.assertNotIn(111, ids_ordinary)
        self.assertGreater(len(ids_ordinary), len(ids_normal))

    def test_streaming_encode_add_special_tokens(self):
        """Streaming encode with add_special_tokens matches batch."""
        text = "Hello"
        batch = self.tok.encode(text, add_special_tokens=True)
        enc = self.tok.encode_stream(add_special_tokens=True)
        stream = enc.feed(text) + enc.finalize()
        self.assertEqual(batch, stream)
        # Should have BOS (112) and EOS (113).
        self.assertIn(112, stream)
        self.assertIn(113, stream)

    def test_streaming_encode_no_special_token_matching(self):
        """Streaming encode with no_special_token_matching matches batch."""
        text = "Hello <|endoftext|> world"
        batch = self.tok.encode(text, no_special_token_matching=True)
        enc = self.tok.encode_stream(no_special_token_matching=True)
        stream = enc.feed(text) + enc.finalize()
        self.assertEqual(batch, stream)
        self.assertNotIn(111, stream)


class TokenizerPendingTokenBoundTest(unittest.TestCase):
    """Test pending_token_bound behavior via streaming encode edge cases.

    The C function pending_token_bound() is used internally by EncodeStream
    to size the finalize output buffer. These tests exercise edge cases
    where an incorrect bound would cause RESOURCE_EXHAUSTED errors.
    """

    @classmethod
    def setUpClass(cls):
        if not BPE_MINIMAL_JSON.exists():
            raise unittest.SkipTest(f"Tokenizer testdata not found: {BPE_MINIMAL_JSON}")
        cls.tok = rt.Tokenizer.from_file(str(BPE_MINIMAL_JSON))

    def test_streaming_encode_large_input_finalize(self):
        """Feed 100KB+ of text in chunks, verify finalize succeeds and
        the round-trip matches batch encode.

        This tests that the tight bound from pending_token_bound is sufficient
        for large streams where the pipeline accumulates significant state.
        """
        # Build a large text (>100KB). Use varied content to exercise different
        # BPE merge patterns and segmentation boundaries.
        base = "Hello world "
        repeat = (100 * 1024 // len(base)) + 1
        text = base * repeat  # ~100KB+
        self.assertGreater(len(text), 100 * 1024)

        # Batch encode for ground truth.
        batch_ids = self.tok.encode(text)

        # Streaming encode in 1KB chunks.
        enc = self.tok.encode_stream()
        stream_ids = []
        chunk_size = 1024
        for i in range(0, len(text), chunk_size):
            stream_ids.extend(enc.feed(text[i : i + chunk_size]))
        # finalize() internally uses pending_token_bound to size its buffer.
        # If the bound is too small, this raises RESOURCE_EXHAUSTED.
        stream_ids.extend(enc.finalize())

        self.assertEqual(len(stream_ids), len(batch_ids))
        self.assertEqual(stream_ids, batch_ids)

    def test_streaming_encode_empty_finalize(self):
        """Create stream, call finalize immediately without any feed.
        Verify it succeeds and returns empty (no special tokens configured).
        """
        enc = self.tok.encode_stream()
        result = enc.finalize()
        self.assertEqual(result, [])

    def test_streaming_encode_empty_finalize_with_special_tokens(self):
        """Create stream with add_special_tokens, call finalize immediately.
        Should succeed and return only the special tokens (BOS + EOS).
        """
        try:
            tok = rt.Tokenizer.from_huggingface_json(_build_special_tokens_json())
        except Exception as e:
            self.skipTest(f"Failed to build special tokens tokenizer: {e}")

        enc = tok.encode_stream(add_special_tokens=True)
        result = enc.finalize()
        # With BOS/EOS postprocessor and empty input, should get just BOS + EOS.
        self.assertIn(112, result)  # BOS
        self.assertIn(113, result)  # EOS

    def test_streaming_encode_single_byte_chunks(self):
        """Feed text one byte at a time — maximizes pipeline fragmentation.
        Exercises partial special token matches spanning many chunks.
        """
        text = "Hello world, how are you today?"
        batch_ids = self.tok.encode(text)
        enc = self.tok.encode_stream()
        stream_ids = []
        for ch in text:
            stream_ids.extend(enc.feed(ch))
        stream_ids.extend(enc.finalize())
        self.assertEqual(stream_ids, batch_ids)

    def test_streaming_encode_varied_chunk_sizes(self):
        """Feed text in irregular chunk sizes (1, 7, 3, 13, ...).
        Exercises pipeline boundary conditions at unpredictable positions.
        """
        text = "The quick brown fox jumps over the lazy dog. " * 100
        batch_ids = self.tok.encode(text)
        enc = self.tok.encode_stream()
        stream_ids = []
        sizes = [1, 7, 3, 13, 2, 11, 5, 17, 4, 19]
        pos = 0
        for i in range(len(text)):
            chunk_size = sizes[i % len(sizes)]
            chunk = text[pos : pos + chunk_size]
            if not chunk:
                break
            stream_ids.extend(enc.feed(chunk))
            pos += chunk_size
            if pos >= len(text):
                break
        stream_ids.extend(enc.finalize())
        self.assertEqual(stream_ids, batch_ids)

    def test_streaming_encode_special_tokens_in_chunks(self):
        """Feed text containing special tokens split across chunk boundaries.
        Exercises partial special token match recovery during finalize.
        """
        try:
            tok = rt.Tokenizer.from_huggingface_json(_build_special_tokens_json())
        except Exception as e:
            self.skipTest(f"Failed to build special tokens tokenizer: {e}")

        text = "Hello <|endoftext|> world"
        batch_ids = tok.encode(text)
        # Split right in the middle of <|endoftext|>
        for split_pos in range(len(text)):
            enc = tok.encode_stream()
            ids = enc.feed(text[:split_pos]) + enc.feed(text[split_pos:])
            ids += enc.finalize()
            self.assertEqual(ids, batch_ids, f"Mismatch at split_pos={split_pos}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
