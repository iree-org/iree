# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
import importlib.util
import pathlib
import re
import sys
import tempfile
import unittest

_MISSING = [
    name
    for name in ("jax", "torch", "torchax")
    if importlib.util.find_spec(name) is None
]
if _MISSING:
    print(f"{', '.join(_MISSING)} not found, skipping torchax_export_test")
    sys.exit(77)

import jax
import numpy as np
import torch
import torchax
from torchax import export as torchax_library

from iree.compiler import tools
from iree_torchax import torchax_export

try:
    from iree import runtime as iree_runtime
except ImportError:
    iree_runtime = None


class _Cache(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(4, 3))
        self.weight = torch.nn.Parameter(torch.arange(3.0) + 1)

    def forward(self, x, position):
        self.cache.index_copy_(0, position, x * self.weight)
        return self.cache.sum(0)


_CALLS = [
    (np.array([[1.0, 2.0, 3.0]], np.float32), np.array([1], np.int64)),
    (np.array([[4.0, 5.0, 6.0]], np.float32), np.array([3], np.int64)),
]


def _lower(module, example):
    exported = torch.export.export(module, example)
    program = torchax_export.exported_program_to_jax(exported)
    lowered = jax.jit(program.func, donate_argnums=(1,), keep_unused=True).lower(
        program.constants,
        program.mutable,
        tuple(np.asarray(a.numpy()) for a in example),
    )
    return program, lowered.as_text()


def _context(vmfb, parameters=None):
    config = iree_runtime.Config("local-task")
    modules = []
    if parameters is not None:
        index = iree_runtime.ParameterIndex()
        index.load(str(parameters))
        modules.append(
            iree_runtime.create_io_parameters_module(
                config.vm_instance, index.create_provider(scope="model")
            )
        )
    vm_module = iree_runtime.VmModule.copy_buffer(config.vm_instance, vmfb)
    modules.append(vm_module)
    context = iree_runtime.SystemContext(vm_modules=modules, config=config)
    return context.modules[vm_module.name].main


def _compile(module, *flags):
    return tools.compile_str(
        module,
        input_type="stablehlo",
        target_backends=["llvm-cpu"],
        extra_args=["--iree-llvmcpu-target-cpu=generic", *flags],
    )


def _cache_eager():
    reference = _Cache()
    with torch.no_grad():
        outputs = [
            reference(*(torch.from_numpy(a) for a in call)).numpy() for call in _CALLS
        ]
    return outputs, reference.cache.detach().numpy()


class _Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.proj = torch.nn.Linear(64, 48)

    def forward(self, x):
        return self.proj(x)


_X = np.random.default_rng(1).standard_normal((5, 64)).astype(np.float32)


class _Scale(torch.nn.Module):
    def forward(self, x, y):
        return x * 2 + y


def _export_batched(batch):
    """`_Scale` exported by torchax, its inputs sharing the dynamic `batch`."""
    exported = torch.export.export(
        _Scale(),
        (torch.randn(5, 7), torch.randn(5, 7)),
        dynamic_shapes=({0: batch}, {0: batch}),
    )
    _, artifact = torchax_library.exported_program_to_stablehlo(exported)
    return artifact.mlir_module()


def _scale_inputs(batch):
    rng = np.random.default_rng(batch)
    return [rng.standard_normal((batch, 7)).astype(np.float32) for _ in range(2)]


class _KvPrefix(torch.nn.Module):
    """Writes q new rows ending at n into two fixed-capacity caches."""

    def __init__(self, capacity=8):
        super().__init__()
        self.register_buffer("k_cache", torch.zeros(1, 1, capacity, 2))
        self.register_buffer("v_cache", torch.zeros(1, 1, capacity, 2))
        self.weight = torch.nn.Parameter(torch.tensor([1.5, -0.5]))

    def forward(self, new_k, new_v, length):
        n = length.shape[0]
        begin = n - new_k.shape[2]
        self.k_cache[:, :, begin:n, :] = new_k * self.weight
        self.v_cache[:, :, begin:n, :] = new_v
        return self.k_cache[:, :, :n, :] + self.v_cache[:, :, :n, :]


class _KvPrefill(_KvPrefix):
    """Writes the first n rows of both caches."""

    def forward(self, new_k, new_v):
        n = new_k.shape[2]
        self.k_cache[:, :, 0:n, :] = new_k * self.weight
        self.v_cache[:, :, 0:n, :] = new_v
        return self.k_cache[:, :, :n, :] + self.v_cache[:, :, :n, :]


def _export_kv_prefix():
    length = torch.export.Dim("n", min=1, max=8)
    # Distinct examples: one tensor passed twice exports as one aliased input.
    new_k, new_v = torch.ones(1, 1, 1, 2), torch.ones(1, 1, 1, 2)
    return torch.export.export(
        _KvPrefix(),
        (new_k, new_v, torch.zeros(3)),
        dynamic_shapes=({}, {}, {0: length}),
    )


class _UnusedConstants(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unused = torch.nn.Parameter(torch.zeros(4, 5))
        self.weight = torch.nn.Parameter(torch.arange(3.0))
        self.register_buffer("table", torch.ones(5, 2))

    def forward(self, x):
        return x * self.weight


class _TorchDtypes(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(12.0).reshape(3, 4))
        self.register_buffer(
            "scale", torch.tensor([[0.5, 2.0], [4.0, 8.0]], dtype=torch.float64)
        )

    def forward(self, x, index):
        return x @ self.weight, index + 1, self.scale * 2


def _session(vmfb):
    config = iree_runtime.Config("local-task")
    vm_module = iree_runtime.VmModule.copy_buffer(config.vm_instance, vmfb)
    context = iree_runtime.SystemContext(vm_modules=[vm_module], config=config)
    return context.modules[vm_module.name].main, config.device


def _rows(count, step):
    rows = np.arange(count * 2, dtype=np.float32).reshape(1, 1, count, 2)
    return rows + 10 * step + 1


_SYSTEMS = np.array(
    [
        [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
        [[2.0, 0.0, 1.0], [0.0, 3.0, 0.0], [1.0, 0.0, 5.0]],
    ],
    np.float32,
)


class _TiedWeights(torch.nn.Module):
    # The embedding and the projection share one weight, as in tied language
    # models.
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(11, 6)
        self.head = torch.nn.Linear(6, 11, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, ids):
        return self.head(self.embed(ids))


class _BufferScale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Buffer(torch.tensor([0.5, 0.25, 0.125]), persistent=False)

    def forward(self, x):
        return x * self.scale


class _ConstantsAndBuffers(torch.nn.Module):
    # Decomposition puts the lifted constant before the submodule's buffer in
    # the program's inputs.
    def __init__(self):
        super().__init__()
        self.inner = _BufferScale()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

    def forward(self, x):
        return self.inner(x) * self.weight + torch.tensor([10.0, 20.0, 30.0])


class _ParameterAndBuffer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.arange(15.0).reshape(3, 5), requires_grad=False
        )
        self.total = torch.nn.Buffer(torch.full((3, 5), 0.5))

    def forward(self, x):
        self.weight.mul_(2.0)
        self.total.add_(x)
        return x * 3.0, {"sum": x + self.weight}


class _MixedConstants(torch.nn.Module):
    # f32 and bf16 weights, an int64 buffer, and a mutated cache.
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(48, 40), requires_grad=False)
        self.narrow = torch.nn.Parameter(
            torch.randn(40, 24).to(torch.bfloat16), requires_grad=False
        )
        self.offsets = torch.nn.Buffer(torch.arange(24, dtype=torch.int64))
        self.cache = torch.nn.Buffer(torch.zeros(5, 24))

    def forward(self, x, position):
        y = (x @ self.weight).to(torch.bfloat16) @ self.narrow
        y = y.to(torch.float32) + self.offsets
        self.cache.index_copy_(0, position, y)
        return self.cache.sum(0)


class _UnalignedTable(torch.nn.Module):
    # A buffer starting one f32 past an aligned allocation, which JAX copies.
    def __init__(self):
        super().__init__()
        self.table = torch.nn.Buffer(torch.randn(1 + 36 * 28)[1:].view(36, 28))

    def forward(self, x):
        return x @ self.table


class _Determinant(torch.nn.Module):
    def forward(self, a):
        return torch.linalg.det(a)


class _Random(torch.nn.Module):
    def __init__(self, draw):
        super().__init__()
        self.draw = draw

    def forward(self, x):
        return self.draw(x)


def _run_with_jax(module, *inputs):
    exported = torch.export.export(module, tuple(map(torch.from_numpy, inputs)))
    program = torchax_export.exported_program_to_jax(exported)
    with jax.enable_x64(True):
        _, outputs = jax.jit(program.func)(program.constants, program.mutable, inputs)
    return [np.asarray(output) for output in outputs]


class TorchaxExportTest(unittest.TestCase):
    def test_cache_update_aliases_the_donated_buffer(self):
        example = tuple(torch.from_numpy(a) for a in _CALLS[0])
        program, module = _lower(_Cache(), example)
        self.assertEqual(program.mutable_names, ["cache"])
        self.assertEqual(program.constant_names, ["weight"])
        signature = re.search(r"func\.func public @main\((.*?)\) ->", module).group(1)
        # Only the cache is donated; the weight and user inputs stay caller-owned.
        self.assertEqual(signature.count("tf.aliasing_output"), 1)
        self.assertRegex(signature, r"tensor<4x3xf32> \{tf\.aliasing_output = 0")

        lowered = tools.compile_str(
            module,
            input_type="stablehlo",
            target_backends=["llvm-cpu"],
            extra_args=["--compile-to=input"],
        ).decode()
        self.assertRegex(lowered, r"tensor<4x3xf32> \{iree\.abi\.output = 0 : index\}")

    @unittest.skipIf(iree_runtime is None, "iree.runtime not available")
    def test_two_calls_match_torch_eager(self):
        expected, final_cache = _cache_eager()
        program, module = _lower(
            _Cache(), tuple(torch.from_numpy(a) for a in _CALLS[0])
        )
        main = _context(_compile(module))
        state = [np.asarray(a) for a in program.mutable]
        initial = state[0].copy()
        for (x, position), want in zip(_CALLS, expected):
            # The export runs JAX in 32-bit mode, which narrows the position.
            *state, output = main(
                *(np.asarray(c) for c in program.constants),
                *state,
                x,
                position.astype(np.int32),
            )
            np.testing.assert_array_equal(output.to_host(), want)
        np.testing.assert_array_equal(state[0].to_host(), final_cache)
        # The caller's host copy is not the device buffer that was donated.
        np.testing.assert_array_equal(np.asarray(program.mutable[0]), initial)

    def test_parameters_leave_the_entry_signature(self):
        program, module = _lower(
            _Cache(), tuple(torch.from_numpy(a) for a in _CALLS[0])
        )
        parameterized = torchax_export.externalize_parameters(
            module, program.constant_names
        )
        self.assertIn(
            'util.global private @__parameter.weight = #flow.parameter.named<"model"::"weight"> : tensor<3xf32>',
            parameterized,
        )
        signature = re.search(r"func\.func @main\((.*?)\) ->", parameterized).group(1)
        self.assertEqual(signature.count("%arg"), 3)
        self.assertRegex(
            signature, r"^%arg0: tensor<4x3xf32> \{tf\.aliasing_output = 0"
        )

    def test_parameters_keep_callers_of_the_entry(self):
        module = """
        func.func @main(%weight: tensor<4xf32>, %x: tensor<4xf32>) -> tensor<4xf32> {
          %0 = arith.addf %weight, %x : tensor<4xf32>
          return %0 : tensor<4xf32>
        }
        func.func @twice(%weight: tensor<4xf32>, %x: tensor<4xf32>) -> tensor<4xf32> {
          %0 = call @main(%weight, %x) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %1 = call @main(%weight, %0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          return %1 : tensor<4xf32>
        }
        """
        parameterized = torchax_export.externalize_parameters(module, ["weight"])
        # Callers keep the full signature; only the public entry loses it.
        self.assertEqual(
            parameterized.count("call @main$with_parameters(%arg0, %arg1)"), 1
        )
        self.assertEqual(
            parameterized.count("call @main$with_parameters(%arg0, %0)"), 1
        )
        tools.compile_str(
            parameterized,
            target_backends=["llvm-cpu"],
            extra_args=["--iree-llvmcpu-target-cpu=generic"],
        )

    def test_parameters_avoid_existing_symbols(self):
        module = """
        util.global private @__parameter.weight = dense<1.0> : tensor<4xf32>
        func.func private @main$with_parameters() -> tensor<4xf32> {
          %0 = util.global.load @__parameter.weight : tensor<4xf32>
          return %0 : tensor<4xf32>
        }
        func.func @main(%weight: tensor<4xf32>, %x: tensor<4xf32>) -> tensor<4xf32> {
          %0 = call @main$with_parameters() : () -> tensor<4xf32>
          %1 = arith.addf %weight, %x : tensor<4xf32>
          %2 = arith.addf %0, %1 : tensor<4xf32>
          return %2 : tensor<4xf32>
        }
        """
        parameterized = torchax_export.externalize_parameters(module, ["weight"])
        self.assertIn(
            "util.global private @__parameter.weight = dense<1.000000e+00>",
            parameterized,
        )
        self.assertIn("func.func private @main$with_parameters() ->", parameterized)
        # The archive key is the parameter name, whatever the global is called.
        loaded = re.search(
            r'util\.global private @(\S+) = #flow\.parameter\.named<"model"::"weight">',
            parameterized,
        ).group(1)
        self.assertNotEqual(loaded, "__parameter.weight")
        self.assertIn(f"util.global.load @{loaded}", parameterized)
        tools.compile_str(
            parameterized,
            target_backends=["llvm-cpu"],
            extra_args=["--iree-llvmcpu-target-cpu=generic"],
        )

    def test_tied_parameters_preserve_bindings_and_cache_alias(self):
        class TiedCache(_Cache):
            def __init__(self):
                super().__init__()
                self.weight_alias = self.weight
                self.unused = torch.nn.Parameter(torch.full((2, 2), 7.0))

            def forward(self, x, position):
                self.cache.index_copy_(0, position, x * self.weight_alias)
                return self.cache.sum(0)

        example = tuple(torch.from_numpy(a) for a in _CALLS[0])
        program, module = _lower(TiedCache(), example)
        self.assertEqual(program.constant_names, ["weight", "unused"])
        np.testing.assert_array_equal(np.asarray(program.constants[0]), [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(
            np.asarray(program.constants[1]), np.full((2, 2), 7.0)
        )
        # The unused parameter must retain its slot so parameter binding cannot
        # consume the cache; tied placeholders share their one constant.
        default_lowering = (
            jax.jit(program.func, donate_argnums=(1,))
            .lower(
                program.constants, program.mutable, tuple(a.numpy() for a in example)
            )
            .as_text()
        )
        default_signature = re.search(
            r"func\.func public @main\((.*?)\) ->", default_lowering
        ).group(1)
        self.assertNotIn("tensor<2x2xf32>", default_signature)
        for exported_module in (
            module,
            str(
                torchax_export.exported_program_to_stablehlo(
                    torch.export.export(TiedCache(), example)
                )[1].mlir_module()
            ),
        ):
            parameterized = torchax_export.externalize_parameters(
                exported_module, program.constant_names
            )
            for name, type_ in zip(
                program.constant_names, ("tensor<3xf32>", "tensor<2x2xf32>")
            ):
                self.assertIn(
                    f"util.global private @__parameter.{name} = "
                    f'#flow.parameter.named<"model"::"{name}"> : {type_}',
                    parameterized,
                )
            signature = re.search(r"func\.func @main\((.*?)\) ->", parameterized).group(
                1
            )
            self.assertEqual(signature.count("%arg"), 3)
            self.assertRegex(
                signature,
                r"^%arg0: tensor<4x3xf32> \{tf\.aliasing_output = 0",
            )

    @unittest.skipIf(iree_runtime is None, "iree.runtime not available")
    def test_cache_with_parameters_matches_torch_eager(self):
        expected, final_cache = _cache_eager()
        program, module = _lower(
            _Cache(), tuple(torch.from_numpy(a) for a in _CALLS[0])
        )
        with tempfile.TemporaryDirectory() as directory:
            archive = pathlib.Path(directory) / "weights.irpa"
            iree_runtime.save_archive_file(
                dict(zip(program.constant_names, map(np.asarray, program.constants))),
                archive,
            )
            main = _context(
                _compile(
                    torchax_export.externalize_parameters(
                        module, program.constant_names
                    )
                ),
                archive,
            )
            state = [np.asarray(a) for a in program.mutable]
            for (x, position), want in zip(_CALLS, expected):
                *state, output = main(*state, x, position.astype(np.int32))
                np.testing.assert_array_equal(output.to_host(), want)
            np.testing.assert_array_equal(state[0].to_host(), final_cache)

    @unittest.skipIf(iree_runtime is None, "iree.runtime not available")
    def test_parameters_match_the_argument_baseline(self):
        program, module = _lower(_Linear(), (torch.from_numpy(_X),))
        parameterized = torchax_export.externalize_parameters(
            module, program.constant_names
        )
        weights = dict(zip(program.constant_names, map(np.asarray, program.constants)))
        baseline = _context(_compile(module))(*weights.values(), _X).to_host()
        with tempfile.TemporaryDirectory() as directory:
            archive = pathlib.Path(directory) / "weights.irpa"
            iree_runtime.save_archive_file(weights, archive)
            result = _context(_compile(parameterized), archive)(_X).to_host()
            np.testing.assert_array_equal(result, baseline)

            missing = pathlib.Path(directory) / "missing.irpa"
            iree_runtime.save_archive_file(
                {k: v for k, v in weights.items() if k != "proj.weight"}, missing
            )
            with self.assertRaisesRegex(ValueError, "key 'proj.weight'"):
                _context(_compile(parameterized), missing)

    def test_data_tiled_weights_pack_in_an_initializer(self):
        program, module = _lower(_Linear(), (torch.from_numpy(_X),))
        parameterized = torchax_export.externalize_parameters(
            module, program.constant_names
        )
        flow = _compile(
            parameterized, "--iree-opt-data-tiling", "--compile-to=flow"
        ).decode()
        # Flow renames the parameter's global; find it by its parameter.
        weight = re.search(
            r"util\.global private @(\S+) .*= "
            r'#flow\.parameter\.named<"model"::"proj\.weight">',
            flow,
        ).group(1)
        hoisted = re.search(
            r"util\.initializer [^{]*\{(?:(?!util\.initializer).)*?"
            rf"util\.global\.load immutable @{re.escape(weight)} "
            r"(?:(?!util\.initializer).)*?util\.global\.store %\S+, @(\S+)",
            flow,
            re.S,
        )
        self.assertIsNotNone(hoisted, "the weight encoding is not hoisted")
        main = flow[flow.index("util.func public @main") :]
        self.assertIn(f"util.global.load immutable @{hoisted.group(1)}", main)
        self.assertNotIn(f"@{weight} ", main)

    @unittest.skipIf(iree_runtime is None, "iree.runtime not available")
    def test_dimension_constraints_bound_the_batch(self):
        module = _export_batched(torch.export.Dim("b", min=3, max=64))
        lowered = _compile(module, "--compile-to=input").decode()
        self.assertIn("<umin = 3, umax = 64>", lowered)

        main = _context(_compile(module))
        x, y = _scale_inputs(5)
        np.testing.assert_array_equal(main(x, y).to_host(), x * 2 + y)
        with self.assertRaisesRegex(RuntimeError, r"constraint \w+ >= 3"):
            main(*_scale_inputs(2))
        with self.assertRaisesRegex(RuntimeError, r"constraint \w+ <= 64"):
            main(*_scale_inputs(65))

    @unittest.skipIf(iree_runtime is None, "iree.runtime not available")
    def test_unconstrained_batch_accepts_one(self):
        # torch.export's default minimum of 2 is not part of the export.
        module = _export_batched(torch.export.Dim("b"))
        lowered = _compile(module, "--compile-to=input").decode()
        self.assertIn("<umin = 1>", lowered)
        self.assertNotIn("umin = 2", lowered)

        x, y = _scale_inputs(1)
        result = _context(_compile(module))(x, y).to_host()
        np.testing.assert_array_equal(result, x * 2 + y)

    @unittest.skipIf(iree_runtime is None, "iree.runtime not available")
    def test_derived_dimension_ties_to_its_symbol(self):
        class Doubled(torch.nn.Module):
            def forward(self, x, y):
                return torch.cat([x, x]) * y

        batch = torch.export.Dim("b")
        exported = torch.export.export(
            Doubled(),
            (torch.randn(5, 7), torch.randn(10, 7)),
            dynamic_shapes=({0: batch}, {0: 2 * batch}),
        )
        _, artifact = torchax_library.exported_program_to_stablehlo(exported)
        module = artifact.mlir_module()
        lowered = _compile(module, "--compile-to=input").decode()
        self.assertIn("<udiv = 2>", lowered)

        main = _context(_compile(module))
        rng = np.random.default_rng(3)
        x = rng.standard_normal((3, 7)).astype(np.float32)
        y = rng.standard_normal((6, 7)).astype(np.float32)
        np.testing.assert_array_equal(main(x, y).to_host(), np.concatenate([x, x]) * y)
        with self.assertRaisesRegex(RuntimeError, r"specification '2\*\w+'"):
            main(x, rng.standard_normal((7, 7)).astype(np.float32))

    @unittest.skipIf(iree_runtime is None, "iree.runtime not available")
    def test_decoder_block_with_dynamic_sequence_matches_eager(self):
        class Block(torch.nn.Module):
            def __init__(self, width=64, heads=4):
                super().__init__()
                torch.manual_seed(0)
                self.heads = heads
                self.norm1 = torch.nn.LayerNorm(width)
                self.qkv = torch.nn.Linear(width, 3 * width)
                self.proj = torch.nn.Linear(width, width)
                self.norm2 = torch.nn.LayerNorm(width)
                self.up = torch.nn.Linear(width, 4 * width)
                self.down = torch.nn.Linear(4 * width, width)

            def forward(self, x):
                b, s, d = x.shape
                head = d // self.heads
                q, k, v = (
                    t.reshape(b, s, self.heads, head).transpose(1, 2)
                    for t in self.qkv(self.norm1(x)).chunk(3, dim=-1)
                )
                w = torch.softmax(q @ k.transpose(-1, -2) / head**0.5, dim=-1)
                x = x + self.proj((w @ v).transpose(1, 2).reshape(b, s, d))
                return x + self.down(torch.nn.functional.gelu(self.up(self.norm2(x))))

        block = Block()
        exported = torch.export.export(
            block,
            (torch.randn(2, 8, 64),),
            dynamic_shapes=({1: torch.export.Dim("s")},),
        )
        weights, artifact = torchax_library.exported_program_to_stablehlo(exported)
        main = _context(_compile(artifact.mlir_module()))
        rng = np.random.default_rng(4)
        for sequence in (5, 1):
            x = rng.standard_normal((2, sequence, 64)).astype(np.float32)
            with torch.no_grad():
                want = block(torch.from_numpy(x)).numpy()
            got = main(*(np.asarray(w) for w in weights), x).to_host()
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)

    @unittest.skipIf(iree_runtime is None, "iree.runtime not available")
    def test_bounded_batch_gets_a_static_transient_slab(self):
        class Attention(torch.nn.Module):
            def forward(self, x):
                return torch.softmax(x @ x.T, dim=-1) @ x

        exported = torch.export.export(
            Attention(),
            (torch.randn(5, 16),),
            dynamic_shapes=({0: torch.export.Dim("b", min=3, max=64)},),
        )
        _, artifact = torchax_library.exported_program_to_stablehlo(exported)
        module = artifact.mlir_module()
        flag = "--iree-stream-bound-transient-allocations"
        stream = _compile(module, flag, "--compile-to=stream").decode()
        self.assertRegex(stream, r"!stream\.resource<transient>\{%c[0-9]+\}")

        main = _context(_compile(module, flag))
        rng = np.random.default_rng(5)
        for batch in (3, 5, 64):
            x = rng.standard_normal((batch, 16)).astype(np.float32)
            with torch.no_grad():
                want = Attention()(torch.from_numpy(x)).numpy()
            np.testing.assert_allclose(main(x).to_host(), want, rtol=1e-5, atol=1e-5)

    def test_stablehlo_keeps_torch_dtypes_and_matmul_precision(self):
        # Torch's int64 and f64 survive, f32 matmuls ask for full precision,
        # and the caller's JAX configuration is left as it was.
        exported = torch.export.export(
            _TorchDtypes(),
            (torch.ones(2, 3), torch.arange(10, dtype=torch.int64).reshape(2, 5)),
        )
        program, artifact = torchax_export.exported_program_to_stablehlo(exported)
        self.assertEqual(
            [str(aval.dtype) for aval in artifact.in_avals],
            ["float32", "float64", "float32", "int64"],
        )
        self.assertEqual(
            [str(aval.dtype) for aval in artifact.out_avals],
            ["float32", "int64", "float64"],
        )
        self.assertIn("precision = [HIGHEST, HIGHEST]", str(artifact.mlir_module()))
        self.assertFalse(jax.config.jax_enable_x64)
        self.assertIsNone(jax.config.jax_default_matmul_precision)

    def test_stablehlo_keeps_unused_constants(self):
        # main takes every constant, used or not, in the documented order.
        exported = torch.export.export(_UnusedConstants(), (torch.ones(2, 3),))
        program, artifact = torchax_export.exported_program_to_stablehlo(exported)
        self.assertEqual(program.constant_names, ["unused", "weight", "table"])
        self.assertEqual(
            sorted(artifact.module_kept_var_idx), list(range(len(artifact.in_avals)))
        )

    @unittest.skipIf(iree_runtime is None, "iree.runtime not available")
    def test_symbolic_cache_prefix_updates(self):
        program, artifact = torchax_export.exported_program_to_stablehlo(
            _export_kv_prefix()
        )
        self.assertEqual(program.mutable_names, ["k_cache", "v_cache"])
        self.assertEqual(program.constant_names, ["weight"])
        module = str(artifact.mlir_module())
        signature = re.search(r"func\.func public @main\((.*?)\) ->", module).group(1)
        self.assertEqual(signature.count("tf.aliasing_output"), 2)
        self.assertIn("tensor<1x1x?x2xf32>", module)
        parameterized = torchax_export.externalize_parameters(
            module, program.constant_names
        )
        entry = re.search(r"func\.func @main\((.*?)\) ->", parameterized).group(1)
        self.assertEqual(entry.count("tf.aliasing_output"), 2)
        lowered = _compile(module, "--compile-to=input").decode()
        self.assertEqual(lowered.count("iree.abi.output"), 2)
        # The constraints prove each update in bounds: no guard, no fallback.
        self.assertNotIn("scf.if", lowered)

        main, device = _session(_compile(module))
        reference = _KvPrefix()
        state = [
            iree_runtime.asdevicearray(device, np.asarray(a)) for a in program.mutable
        ]
        for step, length in enumerate((2, 3)):
            storage = [cache._buffer_view.get_buffer() for cache in state]
            new_k = _rows(1, step)
            new_v = -new_k
            lengths = np.zeros(length, np.float32)
            *state, output = main(
                *(np.asarray(c) for c in program.constants),
                *state,
                new_k,
                new_v,
                lengths,
            )
            # The updates land in the storage the caller donated.
            self.assertEqual(
                [cache._buffer_view.get_buffer() for cache in state], storage
            )
            with torch.no_grad():
                want = reference(
                    torch.from_numpy(new_k),
                    torch.from_numpy(new_v),
                    torch.from_numpy(lengths),
                )
            np.testing.assert_array_equal(output.to_host(), want.numpy())
            np.testing.assert_array_equal(state[0].to_host(), reference.k_cache.numpy())
            np.testing.assert_array_equal(state[1].to_host(), reference.v_cache.numpy())

    @unittest.skipIf(iree_runtime is None, "iree.runtime not available")
    def test_symbolic_cache_prefill(self):
        batch = torch.export.Dim("b", min=1, max=4)
        exported = torch.export.export(
            _KvPrefill(),
            (torch.ones(1, 1, 4, 2), torch.ones(1, 1, 4, 2)),
            dynamic_shapes=({2: 2 * batch}, {2: 2 * batch}),
        )
        program, artifact = torchax_export.exported_program_to_stablehlo(exported)
        self.assertEqual(program.mutable_names, ["k_cache", "v_cache"])
        module = str(artifact.mlir_module())
        signature = re.search(r"func\.func public @main\((.*?)\) ->", module).group(1)
        self.assertEqual(signature.count("tf.aliasing_output"), 2)
        self.assertIn("tensor<1x1x?x2xf32>", signature)

        main, _ = _session(_compile(module))
        reference = _KvPrefill()
        state = [np.asarray(a) for a in program.mutable]
        for step, rows in enumerate((2, 4)):
            new_k = _rows(rows, step)
            new_v = -new_k
            *state, output = main(
                *(np.asarray(c) for c in program.constants), *state, new_k, new_v
            )
            with torch.no_grad():
                want = reference(torch.from_numpy(new_k), torch.from_numpy(new_v))
            np.testing.assert_array_equal(output.to_host(), want.numpy())
            np.testing.assert_array_equal(state[0].to_host(), reference.k_cache.numpy())
            np.testing.assert_array_equal(state[1].to_host(), reference.v_cache.numpy())

    @unittest.skipIf(iree_runtime is None, "iree.runtime not available")
    def test_symbolic_cache_bound_failure(self):
        program, artifact = torchax_export.exported_program_to_stablehlo(
            _export_kv_prefix()
        )
        main, device = _session(_compile(str(artifact.mlir_module())))
        state = [
            iree_runtime.asdevicearray(device, np.asarray(a)) for a in program.mutable
        ]
        new = _rows(1, 0)
        with self.assertRaisesRegex(RuntimeError, r"\w+ <= 8"):
            main(
                *(np.asarray(c) for c in program.constants),
                *state,
                new,
                new,
                np.zeros(9, np.float32),
            )
        for cache in state:
            np.testing.assert_array_equal(cache.to_host(), np.zeros((1, 1, 8, 2)))

    @unittest.skipIf(iree_runtime is None, "iree.runtime not available")
    def test_retained_state_remains_unchanged_without_donation(self):
        exported = torch.export.export(
            _Cache(), tuple(torch.from_numpy(a) for a in _CALLS[0])
        )
        program, artifact = torchax_export.exported_program_to_stablehlo(
            exported, donate_state=False
        )
        main, device = _session(_compile(str(artifact.mlir_module())))
        state = [
            iree_runtime.asdevicearray(device, np.asarray(a)) for a in program.mutable
        ]
        constants = [np.asarray(a.detach()) for a in program.constants]
        eager = _Cache()
        for call in _CALLS:
            retained = state
            snapshots = [np.array(value.to_host(), copy=True) for value in retained]
            *state, output = main(*constants, *state, *call)
            with torch.no_grad():
                expected = eager(*(torch.from_numpy(a) for a in call)).numpy()
            np.testing.assert_array_equal(output.to_host(), expected)
            np.testing.assert_array_equal(state[0].to_host(), eager.cache.numpy())
            for value, before in zip(retained, snapshots, strict=True):
                # Read storage again, independently of DeviceArray's host cache.
                actual = iree_runtime.DeviceArray(device, value._buffer_view).to_host()
                np.testing.assert_array_equal(actual, before)

    def test_no_mutation_donates_nothing(self):
        class Scale(torch.nn.Module):
            def forward(self, x, position):
                return x * 2

        program, module = _lower(Scale(), tuple(torch.from_numpy(a) for a in _CALLS[0]))
        self.assertEqual(program.mutable, [])
        self.assertNotIn("tf.aliasing_output", module)

    def test_jax_constants_share_torch_storage(self):
        module = _MixedConstants()
        exported = torch.export.export(module, (torch.randn(1, 48), torch.tensor([2])))
        program = torchax_export.exported_program_to_jax(exported)
        self.assertEqual(program.constant_names, ["weight", "narrow", "offsets"])
        for name, value in zip(program.constant_names, program.constants):
            self.assertEqual(
                value.unsafe_buffer_pointer(), getattr(module, name).data_ptr(), name
            )
        # Mutable state is donated, so it gets its own storage.
        self.assertNotEqual(
            program.mutable[0].unsafe_buffer_pointer(), module.cache.data_ptr()
        )

    def test_stablehlo_export_returns_the_program_tensors(self):
        module = _MixedConstants()
        exported = torch.export.export(module, (torch.randn(1, 48), torch.tensor([2])))
        program, artifact = torchax_export.exported_program_to_stablehlo(exported)
        self.assertEqual(program.constant_names, ["weight", "narrow", "offsets"])
        self.assertEqual(program.mutable_names, ["cache"])
        for name, value in zip(
            program.constant_names + program.mutable_names,
            program.constants + program.mutable,
        ):
            self.assertIsInstance(value, torch.Tensor)
            self.assertEqual(value.data_ptr(), getattr(module, name).data_ptr(), name)
        self.assertEqual(
            [str(aval.dtype) for aval in artifact.in_avals[:4]],
            ["float32", "bfloat16", "int64", "float32"],
        )

    def test_stablehlo_export_reads_no_weight_data(self):
        # Weights that were never materialised still export; only their
        # shapes and types reach the trace.
        with torch.device("meta"):
            module = _MixedConstants()
        exported = torch.export.export(
            module,
            (torch.randn(1, 48, device="meta"), torch.tensor([2], device="meta")),
        )
        program, artifact = torchax_export.exported_program_to_stablehlo(exported)
        self.assertEqual(
            [tuple(aval.shape) for aval in artifact.in_avals[:4]],
            [(48, 40), (40, 24), (24,), (5, 24)],
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "weight.*has no data"):
                torchax_export.save_parameters(
                    program, pathlib.Path(directory) / "weights.irpa"
                )

    def test_unshareable_constant_requires_opt_in(self):
        module = _UnalignedTable()
        exported = torch.export.export(module, (torch.randn(2, 36),))
        with self.assertRaisesRegex(ValueError, "table.*copy_constants=True"):
            torchax_export.exported_program_to_jax(exported)
        program = torchax_export.exported_program_to_jax(exported, copy_constants=True)
        np.testing.assert_array_equal(
            np.asarray(program.constants[0]), module.table.numpy()
        )
        # Export alone needs no storage it can alias.
        program, _ = torchax_export.exported_program_to_stablehlo(exported)
        self.assertEqual(program.constants[0].data_ptr(), module.table.data_ptr())

    @unittest.skipIf(iree_runtime is None, "iree.runtime not available")
    def test_saved_parameters_match_torch_eager(self):
        module = _MixedConstants()
        example = (torch.randn(1, 48), torch.tensor([2]))
        with torch.no_grad():
            expected = _MixedConstants.forward(copy.deepcopy(module), *example)
        exported = torch.export.export(module, example)
        program, artifact = torchax_export.exported_program_to_stablehlo(exported)
        with tempfile.TemporaryDirectory() as directory:
            archive = pathlib.Path(directory) / "weights.irpa"
            torchax_export.save_parameters(program, archive)
            main = _context(
                _compile(
                    torchax_export.externalize_parameters(
                        str(artifact.mlir_module()), program.constant_names
                    )
                ),
                archive,
            )
            *_, output = main(
                *(np.asarray(a) for a in program.mutable),
                example[0].numpy(),
                example[1].numpy(),
            )
            np.testing.assert_allclose(
                output.to_host(), expected.numpy(), rtol=2e-2, atol=1e-1
            )

    def test_mutated_user_input_is_rejected(self):
        class Increment(torch.nn.Module):
            def forward(self, x):
                x.add_(1)
                return x * 2

        exported = torch.export.export(Increment(), (torch.ones(2, 3),))
        with self.assertRaisesRegex(NotImplementedError, "mutated user inputs"):
            torchax_export.exported_program_to_jax(exported)

    def test_lifted_constants_and_buffers_keep_their_inputs(self):
        x = np.array([1.0, -2.0, 4.0], np.float32)
        (actual,) = _run_with_jax(_ConstantsAndBuffers(), x)
        expected = _ConstantsAndBuffers()(torch.from_numpy(x)).detach().numpy()
        np.testing.assert_allclose(actual, expected)

    def test_mutated_parameter_and_buffer_match_eager(self):
        x = np.linspace(-1.0, 1.0, 15, dtype=np.float32).reshape(3, 5)
        exported = torch.export.export(_ParameterAndBuffer(), (torch.from_numpy(x),))
        program = torchax_export.exported_program_to_jax(exported)
        self.assertEqual(program.mutable_names, ["weight", "total"])
        self.assertEqual(program.constant_names, [])
        new_state, outputs = jax.jit(program.func)(
            program.constants, program.mutable, (x,)
        )
        eager = _ParameterAndBuffer()
        expected = eager(torch.from_numpy(x))
        for actual, want in zip(new_state, (eager.weight, eager.total), strict=True):
            np.testing.assert_allclose(np.asarray(actual), want.detach().numpy())
        for actual, want in zip(
            outputs, (expected[0], expected[1]["sum"]), strict=True
        ):
            np.testing.assert_allclose(np.asarray(actual), want.detach().numpy())

    def test_wrong_input_count_is_named(self):
        x = np.ones((3, 5), np.float32)
        exported = torch.export.export(_ParameterAndBuffer(), (torch.from_numpy(x),))
        program = torchax_export.exported_program_to_jax(exported)
        for inputs in ((), (x, x)):
            with self.subTest(len(inputs)), self.assertRaisesRegex(
                TypeError, f"takes 1 inputs, not {len(inputs)}"
            ):
                program.func(program.constants, program.mutable, inputs)

    def test_tied_weight_is_one_constant(self):
        torch.manual_seed(0)
        module = _TiedWeights()
        ids = np.array([[1, 4, 9], [10, 0, 3]], np.int64)
        exported = torch.export.export(module, (torch.from_numpy(ids),))
        self.assertEqual(
            len([s for s in exported.graph_signature.input_specs if s.target]), 2
        )
        program = torchax_export.exported_program_to_jax(exported)
        self.assertEqual(len(program.constant_names), 1)
        (actual,) = _run_with_jax(module, ids)
        expected = module(torch.from_numpy(ids)).detach().numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_operation_without_lowering_is_named(self):
        with self.assertRaisesRegex(NotImplementedError, r"aten\._linalg_det\.default"):
            _run_with_jax(_Determinant(), _SYSTEMS)

    def test_random_operations_name_the_missing_random_state(self):
        x = np.ones((3, 4), np.float32)
        cases = [
            ("rand", "aten.rand.default", lambda x: x + torch.rand(x.shape)),
            ("randn_like", "aten.randn_like.default", torch.randn_like),
            (
                "dropout",
                "aten.rand.default",
                lambda x: torch.nn.functional.dropout(x, 0.5, training=True),
            ),
        ]
        for name, op, draw in cases:
            with self.subTest(name), self.assertRaisesRegex(
                NotImplementedError, re.escape(op) + ".*random state"
            ):
                _run_with_jax(_Random(draw), x)

    def test_dropout_outside_training_exports(self):
        x = np.arange(12, dtype=np.float32).reshape(3, 4)
        draw = lambda x: torch.nn.functional.dropout(x, 0.5, training=False)
        (result,) = _run_with_jax(_Random(draw), x)
        np.testing.assert_array_equal(result, x)

    def test_torchax_handler_in_torch_runs_in_the_environment(self):
        # Decomposition removes in-place ops, whose Torchax handlers are
        # written in Torch, so only a graph built by hand reaches one.
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.output(graph.call_function(torch.ops.aten.sigmoid_.default, (x,)))
        module = torch.fx.GraphModule(torch.nn.Module(), graph)
        x = np.linspace(-2.0, 2.0, 15, dtype=np.float32).reshape(3, 5)
        result = torchax_export._Interpreter(module, torchax.tensor.Environment()).run(
            jax.numpy.asarray(x), enable_io_processing=False
        )
        expected = torch.sigmoid(torch.from_numpy(x)).numpy()
        np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
