# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checks pretrained SmolLM2 with external weights and functional KV state.

Export runs in a child process to isolate Torchax's process-wide Torch backend
registration. The reference uses the pretrained Transformers decoder layers with
an explicit tensor cache adapter, and checks against ordinary full-prefix Torch
inference. Both registered attention and unsupported-version decomposition paths
are compiled and compared across prefill and repeated decode.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"


class PrefixCache:
    """Adapt Transformers' cache update protocol to registered tensor buffers."""

    def __init__(self, owner, end):
        self.owner = owner
        self.end = end

    def update(self, key, value, layer):
        begin = self.end - key.shape[-2]
        keys = getattr(self.owner, f"key_{layer}")
        values = getattr(self.owner, f"value_{layer}")
        keys[:, :, begin : self.end, :] = key
        values[:, :, begin : self.end, :] = value
        return keys[:, :, : self.end, :], values[:, :, : self.end, :]


class CachedSmolLM2(torch.nn.Module):
    """Use the original decoder layers; only replace Python cache bookkeeping."""

    def __init__(self, model, capacity):
        super().__init__()
        self.model = model
        config = model.config
        shape = (
            1,
            config.num_key_value_heads,
            capacity,
            config.hidden_size // config.num_attention_heads,
        )
        for layer in range(config.num_hidden_layers):
            self.register_buffer(f"key_{layer}", torch.zeros(shape))
            self.register_buffer(f"value_{layer}", torch.zeros(shape))

    def forward(self, tokens, prefix_positions):
        # Positions are explicit, as in Transformers' position_ids API. Their
        # shape also supplies the symbolic prefix length for cache slicing.
        end = prefix_positions.shape[0]
        positions = prefix_positions[end - tokens.shape[1] : end]
        mask = (prefix_positions[None, :] <= positions[:, None])[None, None, :, :]

        hidden = self.model.model.embed_tokens(tokens)
        embeddings = self.model.model.rotary_emb(hidden, positions[None, :])
        cache = PrefixCache(self, end)
        for layer in self.model.model.layers:
            hidden = layer(
                hidden,
                attention_mask=mask,
                past_key_values=cache,
                position_embeddings=embeddings,
            )
        return self.model.lm_head(self.model.model.norm(hidden))


def load_model(args):
    source = args.model_dir or MODEL_ID
    options = {} if args.model_dir else {"revision": MODEL_REVISION}
    model = AutoModelForCausalLM.from_pretrained(
        source, dtype=torch.float32, attn_implementation="sdpa", **options
    ).eval()
    return CachedSmolLM2(model, args.capacity).eval()


def export_programs(args):
    import iree_torchax

    model = load_model(args)
    prefill_length = torch.export.Dim("prefill", min=2, max=args.capacity - 1)
    decode_length = torch.export.Dim("prefix", min=2, max=args.capacity - 1)
    programs = {}
    for name, count, length, shapes in (
        ("prefill", 3, 3, ({1: prefill_length}, {0: prefill_length})),
        ("decode", 1, 4, ({}, {0: decode_length})),
    ):
        print(f"Exporting {name}", flush=True)
        exported = torch.export.export(
            model,
            (
                torch.ones((1, count), dtype=torch.int64),
                torch.arange(length, dtype=torch.int64),
            ),
            dynamic_shapes=shapes,
        )
        program, artifact = iree_torchax.exported_program_to_stablehlo(
            exported, donate_state=not args.retain_state
        )
        module = str(artifact.mlir_module())
        assert 'stablehlo.composite "torch.scaled_dot_product_attention"' in module
        (args.output_dir / f"{name}.export.mlir").write_text(module)
        parameterized = iree_torchax.externalize_parameters(
            module, program.constant_names
        )
        (args.output_dir / f"{name}.mlir").write_text(parameterized)
        programs[name] = program
    first = programs["prefill"]
    assert first.constant_names == programs["decode"].constant_names
    assert first.mutable_names == programs["decode"].mutable_names
    assert len(first.mutable_names) == 2 * model.model.config.num_hidden_layers
    iree_torchax.save_parameters(first, args.output_dir / "weights.irpa")
    with (args.output_dir / "weights.irpa").open("rb") as archive:
        archive_hash = hashlib.file_digest(archive, "sha256").hexdigest()
    metadata = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "dtype": "f32",
        "capacity": args.capacity,
        "state_donated": not args.retain_state,
        "local_model_dir": args.model_dir,
        "parameter_archive_sha256": archive_hash,
        "constant_names": first.constant_names,
        "mutable_names": first.mutable_names,
        "state_shapes": [list(value.shape) for value in first.mutable],
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchax", "jax", "jaxlib", "transformers")
        },
    }
    (args.output_dir / "export.json").write_text(json.dumps(metadata, indent=2) + "\n")


def make_reference(args, metadata):
    model = load_model(args)
    source = args.model_dir or MODEL_ID
    options = {} if args.model_dir else {"revision": MODEL_REVISION}
    tokenizer = AutoTokenizer.from_pretrained(source, **options)
    tokens = tokenizer(args.prompt, return_tensors="pt")["input_ids"]
    if not 2 <= tokens.shape[1] < args.capacity - args.decode_steps:
        raise ValueError("prompt length plus decode steps must fit the symbolic bounds")
    prefix = tokens.clone()
    steps = []
    for step in range(args.decode_steps + 1):
        with torch.no_grad():
            logits = model(tokens, torch.arange(prefix.shape[1], dtype=torch.int64))
            full = model.model(prefix, use_cache=False).logits[:, -tokens.shape[1] :, :]
        np.testing.assert_allclose(logits.numpy(), full.numpy(), rtol=1e-4, atol=1e-4)
        buffers = dict(model.named_buffers())
        steps.append(
            {
                "tokens": tokens.numpy().copy(),
                "length": prefix.shape[1],
                "logits": logits.numpy().copy(),
                "state": [
                    buffers[name].numpy().copy() for name in metadata["mutable_names"]
                ],
            }
        )
        tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        prefix = torch.cat((prefix, tokens), dim=1)
    print("Torch cached reference matches full-prefix inference", flush=True)
    return steps


def unsupported_attention_version(module_text):
    from iree.compiler import ir

    with ir.Context():
        module = ir.Module.parse(module_text)
        changed = 0

        def change(operation):
            nonlocal changed
            if (
                operation.name == "stablehlo.composite"
                and str(operation.attributes["name"])
                == '"torch.scaled_dot_product_attention"'
            ):
                operation.attributes["version"] = ir.IntegerAttr.get(
                    ir.IntegerType.get_signless(32), 2
                )
                changed += 1
            return ir.WalkResult.ADVANCE

        module.operation.walk(change)
        assert changed, "expected an attention composite to exercise fallback"
        return str(module)


def compile_programs(args, fallback):
    from iree.compiler import tools

    mode = "fallback" if fallback else "specialized"
    binaries = {}
    for name in ("prefill", "decode"):
        module = (args.output_dir / f"{name}.mlir").read_text()
        if fallback:
            module = unsupported_attention_version(module)
        flags = (
            ["--iree-llvmcpu-target-cpu=generic"]
            if args.target_backend == "llvm-cpu"
            else []
        ) + args.compiler_arg
        lowered = tools.compile_str(
            module,
            input_type="stablehlo",
            target_backends=[args.target_backend],
            extra_args=[*flags, "--compile-to=input"],
        ).decode()
        assert ("iree_linalg_ext.attention" in lowered) != fallback
        (args.output_dir / f"{mode}-{name}.input.mlir").write_text(lowered)
        print(f"Compiling {mode} {name}", flush=True)
        binaries[name] = tools.compile_str(
            module,
            input_type="stablehlo",
            target_backends=[args.target_backend],
            extra_args=flags,
        )
        (args.output_dir / f"{mode}-{name}.vmfb").write_bytes(binaries[name])
    return binaries


def run_programs(args, metadata, binaries, reference, fallback):
    import iree.runtime as runtime

    config = runtime.Config(args.device)
    index = runtime.ParameterIndex()
    index.load(str(args.output_dir / "weights.irpa"))
    parameters = runtime.create_io_parameters_module(
        config.vm_instance, index.create_provider(scope="model")
    )
    mains, contexts = {}, []
    for name, binary in binaries.items():
        module = runtime.VmModule.copy_buffer(config.vm_instance, binary)
        context = runtime.SystemContext(vm_modules=[parameters, module], config=config)
        contexts.append(context)
        mains[name] = context.modules[module.name].main
    state = [
        runtime.asdevicearray(config.device, np.zeros(shape, np.float32))
        for shape in metadata["state_shapes"]
    ]
    results = []
    for step, expected in enumerate(reference):
        name = "prefill" if step == 0 else "decode"
        retained = state if args.retain_state else []
        snapshots = [value.to_host().copy() for value in retained]
        *state, logits = mains[name](
            *state, expected["tokens"], np.arange(expected["length"], dtype=np.int64)
        )
        for value, before in zip(retained, snapshots, strict=True):
            # Read device storage independently of the previous host mapping.
            actual = runtime.DeviceArray(config.device, value._buffer_view).to_host()
            np.testing.assert_array_equal(actual, before)
        actual_logits = logits.to_host()
        np.testing.assert_allclose(
            actual_logits, expected["logits"], rtol=1e-4, atol=1e-4
        )
        max_cache_error = 0.0
        for actual, want in zip(state, expected["state"], strict=True):
            actual_value = actual.to_host()
            np.testing.assert_allclose(actual_value, want, rtol=1e-4, atol=1e-4)
            max_cache_error = max(
                max_cache_error, float(np.max(np.abs(actual_value - want)))
            )
        predicted = int(actual_logits[0, -1].argmax())
        assert predicted == int(expected["logits"][0, -1].argmax())
        result = {
            "stage": name,
            "prefix_length": expected["length"],
            "next_token": predicted,
            "retained_state_checked": args.retain_state,
            "max_logit_error": float(
                np.max(np.abs(actual_logits - expected["logits"]))
            ),
            "max_cache_error": max_cache_error,
        }
        print(("fallback" if fallback else "specialized"), result, flush=True)
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", help="local snapshot at the pinned revision")
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--export-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--reuse-export", action="store_true")
    parser.add_argument(
        "--retain-state",
        action="store_true",
        help="disable donation and check every previous cache value",
    )
    parser.add_argument(
        "--target-backend",
        choices=("llvm-cpu", "vmvx", "cuda", "rocm", "vulkan-spirv", "metal-spirv"),
        default="llvm-cpu",
    )
    parser.add_argument(
        "--device", help="runtime device URI; defaults to the backend's driver"
    )
    parser.add_argument(
        "--compiler-arg",
        action="append",
        default=[],
        help="additional compiler argument; repeat as needed",
    )
    args = parser.parse_args()
    if args.capacity < 5 or args.decode_steps < 1:
        parser.error("capacity must be at least 5 and decode-steps must be positive")
    if args.device is None:
        args.device = {
            "llvm-cpu": "local-task",
            "vmvx": "local-task",
            "cuda": "cuda",
            "rocm": "hip",
            "vulkan-spirv": "vulkan",
            "metal-spirv": "metal",
        }[args.target_backend]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if args.export_only:
        export_programs(args)
        return
    if not args.reuse_export:
        subprocess.run(
            [sys.executable, __file__, *sys.argv[1:], "--export-only"], check=True
        )
    metadata = json.loads((args.output_dir / "export.json").read_text())
    if metadata["capacity"] != args.capacity or metadata["dtype"] != "f32":
        parser.error("reused export must match capacity and f32 profile")
    if metadata.get("state_donated", True) == args.retain_state:
        parser.error("reused export must match --retain-state")
    reference = make_reference(args, metadata)
    gc.collect()
    results = {
        "export": metadata,
        "prompt": args.prompt,
        "backend": args.target_backend,
        "compiler_args": (
            ["--iree-llvmcpu-target-cpu=generic"]
            if args.target_backend == "llvm-cpu"
            else []
        )
        + args.compiler_arg,
        "device": args.device,
        "rtol": 1e-4,
        "atol": 1e-4,
        "reference_checked_against_full_prefix": True,
        "modes": {},
    }
    for fallback in (False, True):
        binaries = compile_programs(args, fallback)
        mode = "fallback" if fallback else "specialized"
        results["modes"][mode] = run_programs(
            args, metadata, binaries, reference, fallback
        )
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
