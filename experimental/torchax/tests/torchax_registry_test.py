# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib.util
import subprocess
import sys
import textwrap
import unittest

_MISSING = [
    name
    for name in ("jax", "torch", "torchax")
    if importlib.util.find_spec(name) is None
]
if _MISSING:
    print(f"{', '.join(_MISSING)} not found, skipping torchax_registry_test")
    sys.exit(77)

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchax
from torchax.ops import ops_registry

from iree.compiler import tools
from iree_torchax import _torchax_registry
import iree_torchax

try:
    from iree import runtime as iree_runtime
except ImportError:
    iree_runtime = None


@torch.library.custom_op("iree_registry_test::conflict", mutates_args=())
def _conflict(x: torch.Tensor) -> torch.Tensor:
    return x * 2


@torch.library.custom_op("iree_registry_test::foreign", mutates_args=())
def _foreign(x: torch.Tensor) -> torch.Tensor:
    return x * 2


@torch.library.custom_op("iree_registry_test::packeted", mutates_args=())
def _packeted(x: torch.Tensor) -> torch.Tensor:
    return x * 2


@torch.library.custom_op("iree_registry_test::decomposed", mutates_args=())
def _decomposed(x: torch.Tensor) -> torch.Tensor:
    return x * 2


@torch.library.custom_op("iree_registry_test::twice", mutates_args=())
def _twice(x: torch.Tensor) -> torch.Tensor:
    return x * 2


@_twice.register_fake
def _(x):
    return torch.empty_like(x)


@torch.library.custom_op("iree_registry_test::snap", mutates_args=())
def _snap(x: torch.Tensor) -> torch.Tensor:
    return x * 2


@_snap.register_fake
def _(x):
    return torch.empty_like(x)


class _Snap(torch.nn.Module):
    def forward(self, x):
        return torch.ops.iree_registry_test.snap(x)


def _scaling(factor):
    def handler(x, *, env):
        return env.j2t_iso(x.jax() * factor)

    return handler


def _state(op):
    """Everything a registration of `op` may change."""
    entries = {entry.key: entry for entry in _torchax_registry.catalog()}
    return (
        entries.get(op),
        ops_registry.all_aten_ops.get(op),
        torchax.default_env()._ops.get(op),
    )


def _run(op):
    """`op` in a new environment given IREE's registrations."""
    env = torchax.tensor.Environment()
    iree_torchax.install(env)
    x = env.j2t_iso(jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3))
    with env:
        return np.asarray(op(x).jax())


class RegistrationConflictsTest(unittest.TestCase):
    def test_registration_conflicts(self):
        op = torch.ops.iree_registry_test.conflict.default
        double = _scaling(2.0)
        iree_torchax.register_composite(op, double, provider="example.a")
        registered = _state(op)
        # Torchax's registry and default environment stay as they were.
        self.assertEqual(registered[1:], (None, None))
        np.testing.assert_array_equal(_run(op), np.arange(6.0).reshape(2, 3) * 2)

        # The identical registration changes nothing.
        iree_torchax.register_composite(op, double, provider="example.a")
        self.assertEqual(_state(op), registered)

        triple = _scaling(3.0)
        with self.assertRaisesRegex(ValueError, "example.a.*example.b"):
            iree_torchax.register_composite(op, triple, provider="example.b")
        self.assertEqual(_state(op), registered)
        with self.assertRaisesRegex(ValueError, "conflict"):
            iree_torchax.register_composite(op, triple, provider="example.a")
        self.assertEqual(_state(op), registered)

        # A composite never becomes a decomposition, replacement or not.
        for replace in (False, True):
            with self.assertRaisesRegex(ValueError, "composite"):
                iree_torchax.register_decomposition(
                    op, lambda x: x * 3, provider="example.a", replace=replace
                )
            self.assertEqual(_state(op), registered)

        with self.assertRaises(TypeError):
            iree_torchax.register_composite(
                torch.ops.iree_registry_test.conflict, triple, provider="example.b"
            )
        with self.assertRaises(TypeError):
            iree_torchax.register_composite(op, None, provider="example.b")
        with self.assertRaises(ValueError):
            iree_torchax.register_composite(op, triple, provider="")
        self.assertEqual(_state(op), registered)

        iree_torchax.register_composite(op, triple, provider="example.b", replace=True)
        self.assertEqual(_state(op)[0].provider, "example.b")
        np.testing.assert_array_equal(_run(op), np.arange(6.0).reshape(2, 3) * 3)

    def test_foreign_handler_stays(self):
        # A user's Torchax handlers win over IREE's in every environment.
        op = torch.ops.iree_registry_test.foreign.default

        def users(x):
            return x

        torchax.default_env().override_op_definition(op, users)
        ops_registry.register_torch_dispatch_op(
            op,
            _scaling(7.0),
            is_jax_function=False,
            is_user_defined=True,
            needs_env=True,
        )
        for replace in (False, True):
            iree_torchax.register_composite(
                op, _scaling(4.0), provider="example.a", replace=replace
            )
            self.assertIs(torchax.default_env()._ops[op].func, users)
            np.testing.assert_array_equal(_run(op), np.arange(6.0).reshape(2, 3) * 7)

    def test_packet_handler_conflicts(self):
        # A user's handler for the whole packet covers its overloads.
        packet = torch.ops.iree_registry_test.packeted
        op = packet.default
        ops_registry.register_torch_dispatch_op(
            packet,
            _scaling(5.0),
            is_jax_function=False,
            is_user_defined=True,
            needs_env=True,
        )
        before = ops_registry.all_aten_ops.get(packet)
        iree_torchax.register_composite(op, _scaling(4.0), provider="example.a")
        self.assertIs(ops_registry.all_aten_ops.get(packet), before)
        np.testing.assert_array_equal(_run(op), np.arange(6.0).reshape(2, 3) * 5)

    def test_decomposition_conflicts(self):
        op = torch.ops.iree_registry_test.decomposed.default

        def add(x):
            return x + x

        iree_torchax.register_decomposition(op, add, provider="example.a")
        registered = _state(op)
        iree_torchax.register_decomposition(op, add, provider="example.a")
        self.assertEqual(_state(op), registered)
        with self.assertRaisesRegex(ValueError, "example.a.*example.b"):
            iree_torchax.register_decomposition(
                op, lambda x: x * 2, provider="example.b"
            )
        self.assertEqual(_state(op), registered)
        with self.assertRaisesRegex(ValueError, "decomposition"):
            iree_torchax.register_composite(
                op, _scaling(2.0), provider="example.a", replace=True
            )
        self.assertEqual(_state(op), registered)


class _ClipTwice(torch.nn.Module):
    def forward(self, x):
        clipped = torch.nn.functional.hardtanh(x, -0.5, 0.5)
        return torch.ops.iree_registry_test.twice(clipped)


def _main(vmfb):
    config = iree_runtime.Config("local-task")
    vm_module = iree_runtime.VmModule.copy_buffer(config.vm_instance, vmfb)
    context = iree_runtime.SystemContext(vm_modules=[vm_module], config=config)
    return context.modules[vm_module.name].main


def _compile(text):
    return tools.compile_str(
        text,
        input_type="stablehlo",
        target_backends=["llvm-cpu"],
        extra_args=["--iree-llvmcpu-target-cpu=generic"],
    )


class RegisteredDecompositionTest(unittest.TestCase):
    def test_registered_decomposition(self):
        # The plugin's hardtanh selects; the default rule takes max and min.
        def by_select(x, min_val=-1.0, max_val=1.0):
            return torch.where(
                x < min_val, min_val, torch.where(x > max_val, max_val, x)
            )

        iree_torchax.register_decomposition(
            torch.ops.aten.hardtanh.default, by_select, provider="example.ops"
        )
        iree_torchax.register_decomposition(
            torch.ops.iree_registry_test.twice.default,
            lambda x: x + x,
            provider="example.ops",
        )
        x = torch.linspace(-1.0, 1.0, 6).reshape(2, 3)
        exported = torch.export.export(_ClipTwice(), (x,))
        _, artifact = iree_torchax.exported_program_to_stablehlo(exported)
        text = str(artifact.mlir_module())
        self.assertIn("stablehlo.select", text)
        self.assertNotIn("stablehlo.maximum", text)
        self.assertNotIn("iree_registry_test", text)
        self.assertNotIn("stablehlo.composite", text)
        self.assertNotIn("custom_call", text)
        if iree_runtime is None:
            return
        result = _main(_compile(text))(x.numpy())
        np.testing.assert_allclose(
            np.asarray(result), np.clip(x.numpy(), -0.5, 0.5) * 2
        )


def _traced(program, x):
    _, outputs = jax.jit(program.func)(program.constants, program.mutable, (x.numpy(),))
    return np.asarray(outputs[0])


class ExportSnapshotTest(unittest.TestCase):
    def test_export_snapshot_and_override(self):
        op = torch.ops.iree_registry_test.snap.default
        x = torch.arange(6.0).reshape(2, 3)
        exported = torch.export.export(_Snap(), (x,))
        iree_torchax.register_composite(
            op,
            _scaling(2.0),
            provider="example.snap",
            export_handler=_scaling(5.0),
        )
        np.testing.assert_array_equal(_run(op), x.numpy() * 2)

        # Prepared, then traced only after the replacement.
        prepared = iree_torchax.exported_program_to_jax(exported)
        iree_torchax.register_composite(
            op, _scaling(3.0), provider="example.snap", replace=True
        )
        np.testing.assert_array_equal(_traced(prepared, x), x.numpy() * 5)
        later = iree_torchax.exported_program_to_jax(exported)
        np.testing.assert_array_equal(_traced(later, x), x.numpy() * 3)

        # A user's Torchax handler, inherited by the export environment, stays.
        ops_registry.register_torch_dispatch_op(
            op,
            _scaling(7.0),
            is_jax_function=False,
            is_user_defined=True,
            needs_env=True,
        )
        overridden = iree_torchax.exported_program_to_jax(exported)
        np.testing.assert_array_equal(_traced(overridden, x), x.numpy() * 7)


class CompositeArgumentsTest(unittest.TestCase):
    def test_composite_arguments(self):
        def body(x):
            return x

        with self.assertRaises(TypeError):
            iree_torchax.composite(None, name="example.body")
        with self.assertRaises(ValueError):
            iree_torchax.composite(body, name="")
        with self.assertRaises(TypeError):
            iree_torchax.composite(body, name="example.body", version=1.0)
        for version in (-1, 2**31):
            with self.assertRaises(ValueError):
                iree_torchax.composite(body, name="example.body", version=version)


class PluginRegistrationTest(unittest.TestCase):
    def test_prepared_program_ignores_later_packet_registration(self):
        # Torchax registers gelu by packet, which a later overload plugin hides.
        script = textwrap.dedent(
            """
            import jax
            import numpy as np
            import torch
            import iree_torchax

            class Gelu(torch.nn.Module):
                def forward(self, x):
                    return torch.nn.functional.gelu(x)

            x = torch.linspace(-2.0, 2.0, 6).reshape(2, 3)
            prepared = iree_torchax.exported_program_to_jax(
                torch.export.export(Gelu(), (x,)))

            def zeros(x, *, env):
                return env.j2t_iso(x.jax() * 0)

            iree_torchax.register_composite(
                torch.ops.aten.gelu.default, zeros, provider="example.ops")
            _, outputs = jax.jit(prepared.func)(
                prepared.constants, prepared.mutable, (x.numpy(),))
            np.testing.assert_allclose(
                np.asarray(outputs[0]), torch.nn.functional.gelu(x).numpy(),
                rtol=1e-6, atol=1e-6)
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_plugin_registration_after_import(self):
        # A fresh process, so `old` predates IREE's registrations.
        script = textwrap.dedent(
            """
            import jax
            import jax.numpy as jnp
            import numpy as np
            import torch
            import torchax

            old = torchax.tensor.Environment()
            from iree.compiler import tools
            import iree_torchax

            @torch.library.custom_op("iree_plugin_test::scale", mutates_args=())
            def scale(x: torch.Tensor) -> torch.Tensor:
                return x * 2

            @scale.register_fake
            def _(x):
                return torch.empty_like(x)

            scale_composite = iree_torchax.composite(
                lambda x: x * 2, name="example.scale"
            )

            def handler(x, *, env):
                return env.j2t_iso(scale_composite(x.jax()))

            op = torch.ops.iree_plugin_test.scale.default
            iree_torchax.register_composite(op, handler, provider="example.ops")
            new = torchax.tensor.Environment()

            def run(env):
                x = env.j2t_iso(jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3))
                with env:
                    return np.asarray(op(x).jax())

            expected = np.arange(6.0).reshape(2, 3) * 2
            # Registration reaches no environment by itself.
            for env in (torchax.default_env(), old, new):
                assert env._ops.get(op) is None
            for env in (old, new):
                iree_torchax.install(env)
                np.testing.assert_array_equal(run(env), expected)

            class Scale(torch.nn.Module):
                def forward(self, x):
                    return torch.ops.iree_plugin_test.scale(x)

            x = torch.arange(6.0).reshape(2, 3)
            _, artifact = iree_torchax.exported_program_to_stablehlo(
                torch.export.export(Scale(), (x,)))
            text = str(artifact.mlir_module())
            assert 'stablehlo.composite "example.scale"' in text, text
            try:
                from iree import runtime as iree_runtime
            except ImportError:
                raise SystemExit(0)
            vmfb = tools.compile_str(
                text,
                input_type="stablehlo",
                target_backends=["llvm-cpu"],
                extra_args=["--iree-llvmcpu-target-cpu=generic"],
            )
            config = iree_runtime.Config("local-task")
            module = iree_runtime.VmModule.copy_buffer(config.vm_instance, vmfb)
            context = iree_runtime.SystemContext(vm_modules=[module], config=config)
            result = context.modules[module.name].main(x.numpy())
            np.testing.assert_array_equal(np.asarray(result), expected)
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
