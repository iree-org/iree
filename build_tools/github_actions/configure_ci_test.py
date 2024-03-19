#!/usr/bin/env python3

# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pathlib
import unittest

import configure_ci

SORTED_DEFAULT_BENCHMARK_PRESETS_STR = ",".join(
    sorted(configure_ci.DEFAULT_BENCHMARK_PRESET_GROUP)
)


class ConfigureCITest(unittest.TestCase):
    def test_get_benchmark_presets_no_preset(self):
        presets_str = configure_ci.get_benchmark_presets(
            trailers={},
            labels=["unrelated-labels"],
            is_pr=True,
            is_llvm_integrate_pr=False,
        )

        self.assertEqual(presets_str, "")

    def test_get_benchmark_presets_from_pr_labels(self):
        presets_str = configure_ci.get_benchmark_presets(
            trailers={},
            labels=["benchmarks:x86_64", "benchmarks:cuda"],
            is_pr=True,
            is_llvm_integrate_pr=False,
        )

        self.assertEqual(presets_str, "comp-stats,cuda,x86_64")

    def test_get_benchmark_presets_from_trailers_and_labels(self):
        presets_str = configure_ci.get_benchmark_presets(
            trailers={"benchmark-extra": "android-cpu,cuda-large,x86_64-large"},
            labels=["benchmarks:vulkan-nvidia"],
            is_pr=True,
            is_llvm_integrate_pr=False,
        )

        self.assertEqual(
            presets_str, "android-cpu,comp-stats,cuda-large,vulkan-nvidia,x86_64-large"
        )

    def test_get_benchmark_presets_from_default_group(self):
        presets_str = configure_ci.get_benchmark_presets(
            trailers={"benchmark-extra": "default"},
            labels=[],
            is_pr=True,
            is_llvm_integrate_pr=False,
        )

        self.assertEqual(presets_str, SORTED_DEFAULT_BENCHMARK_PRESETS_STR)
        # Sanity check to ensure no `*-large` preset in the default group.
        self.assertNotIn("-large", presets_str)

    def test_get_benchmark_presets_for_non_pr(self):
        presets_str = configure_ci.get_benchmark_presets(
            trailers={}, labels=[], is_pr=False, is_llvm_integrate_pr=False
        )

        self.assertEqual(presets_str, SORTED_DEFAULT_BENCHMARK_PRESETS_STR)

    def test_get_benchmark_presets_for_llvm_integrate_pr(self):
        presets_str = configure_ci.get_benchmark_presets(
            trailers={}, labels=[], is_pr=True, is_llvm_integrate_pr=True
        )

        self.assertEqual(presets_str, SORTED_DEFAULT_BENCHMARK_PRESETS_STR)

    # Sample PR description:
    # ```
    # PR Title
    #
    # PR body...
    #
    # skip-llvm-integrate-benchmark: some good reasons
    # ```
    # Result: No benchmark is automatically enabled on the LLVM integrate PR.
    def test_get_benchmark_presets_skip_llvm_integrate_benchmark(self):
        presets_str = configure_ci.get_benchmark_presets(
            trailers={"skip-llvm-integrate-benchmark": "some good reasons"},
            labels=[],
            is_pr=True,
            is_llvm_integrate_pr=True,
        )

        self.assertEqual(presets_str, "")

    def test_get_benchmark_presets_unknown_preset(self):
        self.assertRaises(
            ValueError,
            lambda: configure_ci.get_benchmark_presets(
                trailers={"benchmark-extra": "unknown"},
                labels=[],
                is_pr=True,
                is_llvm_integrate_pr=False,
            ),
        )

    def test_parse_jobs_trailer(self):
        trailers = {"key": "job1,job2"}
        key = "key"
        all_jobs = {"job1", "job2", "job3"}
        jobs = configure_ci.parse_jobs_trailer(trailers, key, all_jobs)
        self.assertCountEqual(jobs, {"job1", "job2"})

    def test_parse_jobs_trailer_whitespace(self):
        trailers = {"key": "  job1 ,  job2 "}
        key = "key"
        all_jobs = {"job1", "job2", "job3"}
        jobs = configure_ci.parse_jobs_trailer(trailers, key, all_jobs)
        self.assertCountEqual(jobs, {"job1", "job2"})

    def test_parse_jobs_trailer_all_with_others(self):
        bad_text = "job1, all"
        trailers = {"key": bad_text}
        key = "key"
        all_jobs = {"job1", "job2", "job3"}
        with self.assertRaises(ValueError) as cm:
            configure_ci.parse_jobs_trailer(trailers, key, all_jobs)

        msg = str(cm.exception)
        self.assertIn(configure_ci.ALL_KEY, msg)
        self.assertIn(bad_text, msg)

    def test_parse_jobs_unknown_job(self):
        unknown_job = "unknown_job"
        trailers = {"key": f"job1, {unknown_job}"}
        key = "key"
        all_jobs = {"job1", "job2", "job3"}
        # Unknown jobs log a warning, as multiple workflows use configure_ci
        # and a name may be recognized by one workflow and not another.
        jobs = configure_ci.parse_jobs_trailer(trailers, key, all_jobs)
        self.assertCountEqual(jobs, {"job1"})

    def test_get_enabled_jobs_all(self):
        trailers = {}
        all_jobs = {"job1", "job2", "job3"}
        is_pr = True
        is_llvm_integrate_pr = False
        modified_paths = ["runtime/file"]
        jobs = configure_ci.get_enabled_jobs(
            trailers,
            all_jobs,
            modified_paths=modified_paths,
            is_pr=is_pr,
            is_llvm_integrate_pr=is_llvm_integrate_pr,
        )
        self.assertCountEqual(jobs, all_jobs)

    def test_get_enabled_jobs_postsubmit(self):
        trailers = {}
        default_jobs = {"job1", "job2", "job3"}
        postsubmit_job = next(iter(configure_ci.DEFAULT_POSTSUBMIT_ONLY_JOBS))
        all_jobs = default_jobs | {postsubmit_job}
        is_pr = False
        is_llvm_integrate_pr = False
        modified_paths = ["runtime/file"]
        jobs = configure_ci.get_enabled_jobs(
            trailers,
            all_jobs,
            modified_paths=modified_paths,
            is_pr=is_pr,
            is_llvm_integrate_pr=is_llvm_integrate_pr,
        )
        self.assertCountEqual(jobs, all_jobs)

    def test_get_enabled_jobs_no_postsubmit(self):
        trailers = {}
        default_jobs = {"job1", "job2", "job3"}
        postsubmit_job = next(iter(configure_ci.DEFAULT_POSTSUBMIT_ONLY_JOBS))
        all_jobs = default_jobs | {postsubmit_job}
        is_pr = True
        is_llvm_integrate_pr = False
        modified_paths = ["runtime/file"]
        jobs = configure_ci.get_enabled_jobs(
            trailers,
            all_jobs,
            modified_paths=modified_paths,
            is_pr=is_pr,
            is_llvm_integrate_pr=is_llvm_integrate_pr,
        )
        self.assertCountEqual(jobs, default_jobs)

    def test_get_enabled_jobs_llvm_integrate(self):
        trailers = {}
        default_jobs = {"job1", "job2", "job3"}
        postsubmit_job = next(iter(configure_ci.DEFAULT_POSTSUBMIT_ONLY_JOBS))
        all_jobs = default_jobs | {postsubmit_job}
        is_pr = True
        is_llvm_integrate_pr = True
        modified_paths = ["runtime/file"]
        jobs = configure_ci.get_enabled_jobs(
            trailers,
            all_jobs,
            modified_paths=modified_paths,
            is_pr=is_pr,
            is_llvm_integrate_pr=is_llvm_integrate_pr,
        )
        self.assertCountEqual(jobs, all_jobs)

    def test_get_enabled_jobs_no_modifies(self):
        trailers = {}
        default_jobs = {"job1", "job2", "job3"}
        postsubmit_job = next(iter(configure_ci.DEFAULT_POSTSUBMIT_ONLY_JOBS))
        all_jobs = default_jobs | {postsubmit_job}
        is_pr = True
        is_llvm_integrate_pr = False
        modified_paths = ["experimental/file"]
        jobs = configure_ci.get_enabled_jobs(
            trailers,
            all_jobs,
            modified_paths=modified_paths,
            is_pr=is_pr,
            is_llvm_integrate_pr=is_llvm_integrate_pr,
        )
        self.assertCountEqual(jobs, {})

    def test_get_enabled_jobs_skip(self):
        trailers = {configure_ci.Trailer.SKIP_JOBS: "job1,job2"}
        default_jobs = {"job1", "job2", "job3"}
        postsubmit_job = next(iter(configure_ci.DEFAULT_POSTSUBMIT_ONLY_JOBS))
        all_jobs = default_jobs | {postsubmit_job}
        is_pr = True
        is_llvm_integrate_pr = False
        modified_paths = ["runtime/file"]
        jobs = configure_ci.get_enabled_jobs(
            trailers,
            all_jobs,
            modified_paths=modified_paths,
            is_pr=is_pr,
            is_llvm_integrate_pr=is_llvm_integrate_pr,
        )
        self.assertCountEqual(jobs, {"job3"})

    def test_get_enabled_jobs_skip_all(self):
        trailers = {configure_ci.Trailer.SKIP_JOBS: "all"}
        default_jobs = {"job1", "job2", "job3"}
        postsubmit_job = next(iter(configure_ci.DEFAULT_POSTSUBMIT_ONLY_JOBS))
        all_jobs = default_jobs | {postsubmit_job}
        is_pr = True
        is_llvm_integrate_pr = False
        modified_paths = ["runtime/file"]
        jobs = configure_ci.get_enabled_jobs(
            trailers,
            all_jobs,
            modified_paths=modified_paths,
            is_pr=is_pr,
            is_llvm_integrate_pr=is_llvm_integrate_pr,
        )
        self.assertCountEqual(jobs, {})

    def test_get_enabled_jobs_extra(self):
        postsubmit_job = next(iter(configure_ci.DEFAULT_POSTSUBMIT_ONLY_JOBS))
        trailers = {configure_ci.Trailer.EXTRA_JOBS: postsubmit_job}
        default_jobs = {"job1", "job2", "job3"}
        all_jobs = default_jobs | {postsubmit_job}
        is_pr = True
        is_llvm_integrate_pr = False
        modified_paths = ["runtime/file"]
        jobs = configure_ci.get_enabled_jobs(
            trailers,
            all_jobs,
            modified_paths=modified_paths,
            is_pr=is_pr,
            is_llvm_integrate_pr=is_llvm_integrate_pr,
        )
        self.assertCountEqual(jobs, all_jobs)

    def test_get_enabled_jobs_exactly(self):
        postsubmit_job = next(iter(configure_ci.DEFAULT_POSTSUBMIT_ONLY_JOBS))
        trailers = {configure_ci.Trailer.EXACTLY_JOBS: postsubmit_job}
        default_jobs = {"job1", "job2", "job3"}
        all_jobs = default_jobs | {postsubmit_job}
        is_pr = True
        is_llvm_integrate_pr = False
        modified_paths = ["runtime/file"]
        jobs = configure_ci.get_enabled_jobs(
            trailers,
            all_jobs,
            modified_paths=modified_paths,
            is_pr=is_pr,
            is_llvm_integrate_pr=is_llvm_integrate_pr,
        )
        self.assertCountEqual(jobs, {postsubmit_job})

    def test_get_enabled_jobs_metal(self):
        trailers = {}
        all_jobs = {"job1"}
        is_pr = True
        is_llvm_integrate_pr = False
        modified_paths = ["runtime/src/iree/hal/drivers/metal/file"]
        jobs = configure_ci.get_enabled_jobs(
            trailers,
            all_jobs,
            modified_paths=modified_paths,
            is_pr=is_pr,
            is_llvm_integrate_pr=is_llvm_integrate_pr,
        )
        expected_jobs = {"job1", "build_test_all_macos_arm64"}
        self.assertCountEqual(jobs, expected_jobs)

    def test_get_enabled_jobs_windows(self):
        trailers = {}
        all_jobs = {"job1"}
        is_pr = True
        is_llvm_integrate_pr = False
        modified_paths = ["runtime/src/iree/base/internal/threading_win32.c"]
        jobs = configure_ci.get_enabled_jobs(
            trailers,
            all_jobs,
            modified_paths=modified_paths,
            is_pr=is_pr,
            is_llvm_integrate_pr=is_llvm_integrate_pr,
        )
        expected_jobs = {"job1", "build_test_all_windows"}
        self.assertCountEqual(jobs, expected_jobs)

    def test_get_enabled_jobs_windows_docs(self):
        # docs/ directory is excluded from CI, superceding "windows" inclusion
        trailers = {}
        all_jobs = {"job1"}
        is_pr = True
        is_llvm_integrate_pr = False
        modified_paths = ["docs/windows.md"]
        jobs = configure_ci.get_enabled_jobs(
            trailers,
            all_jobs,
            modified_paths=modified_paths,
            is_pr=is_pr,
            is_llvm_integrate_pr=is_llvm_integrate_pr,
        )
        expected_jobs = {}
        self.assertCountEqual(jobs, expected_jobs)

    def test_parse_path_from_workflow_ref(self):
        path = configure_ci.parse_path_from_workflow_ref(
            "octocat/example", "octocat/example/.github/test.yml@1234"
        )

        self.assertEqual(path, pathlib.Path(".github/test.yml"))

    def test_parse_path_from_workflow_ref_invalid_ref(self):
        self.assertRaises(
            ValueError,
            lambda: configure_ci.parse_path_from_workflow_ref(
                "octocat/example", "squid/unknown/.github/test.yml@1234"
            ),
        )


if __name__ == "__main__":
    unittest.main()
