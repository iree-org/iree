#!/usr/bin/env python3

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Determines whether CI should run on a given PR.

The following environment variables are required:
- GITHUB_REPOSITORY: GitHub org and repository, e.g. openxla/iree.
- GITHUB_WORKFLOW_REF: GitHub workflow ref, e.g.
    openxla/iree/.github/workflows/ci.yml@refs/pull/1/merge.
- GITHUB_EVENT_NAME: GitHub event name, e.g. pull_request.
- GITHUB_OUTPUT: path to write workflow output variables.
- GITHUB_STEP_SUMMARY: path to write workflow summary output.

When GITHUB_EVENT_NAME is "pull_request", there are additional environment
variables to be set:
- PR_BRANCH (required): PR source branch.
- PR_TITLE (required): PR title.
- PR_BODY (optional): PR description.
- PR_LABELS (optional): JSON list of PR label names.
- BASE_REF (required): base commit SHA of the PR.
- ORIGINAL_PR_TITLE (optional): PR title from the original PR event, showing a
    notice if PR_TITLE is different.
- ORIGINAL_PR_BODY (optional): PR description from the original PR event,
    showing a notice if PR_BODY is different. ORIGINAL_PR_TITLE must also be
    set.
- ORIGINAL_PR_LABELS (optional): PR labels from the original PR event, showing a
    notice if PR_LABELS is different. ORIGINAL_PR_TITLE must also be set.

Exit code 0 indicates that it should and exit code 2 indicates that it should
not.
"""

import difflib
import enum
import fnmatch
import json
import os
import re
import pathlib
import string
import subprocess
import sys
import textwrap
from typing import Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import yaml

# Add build_tools python dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.with_name("python")))

from benchmark_suites.iree import benchmark_presets


# We don't get StrEnum till Python 3.11
@enum.unique
class Trailer(str, enum.Enum):
    __str__ = str.__str__

    SKIP_CI = "skip-ci"
    SKIP_JOBS = "ci-skip"
    EXTRA_JOBS = "ci-extra"
    EXACTLY_JOBS = "ci-exactly"
    RUNNER_ENV = "runner-env"
    BENCHMARK_EXTRA = "benchmark-extra"
    # Trailer to prevent benchmarks from always running on LLVM integration PRs.
    SKIP_LLVM_INTEGRATE_BENCHMARK = "skip-llvm-integrate-benchmark"

    # Before Python 3.12, it the native __contains__ doesn't work for checking
    # member values like this and it's not possible to easily override this.
    # https://docs.python.org/3/library/enum.html#enum.EnumType.__contains__
    @classmethod
    def contains(cls, val):
        try:
            cls(val)
        except ValueError:
            return False
        return True


# This is to help prevent typos. For now we hard error on any trailer that
# starts with this prefix but isn't in our list. We can add known commonly used
# trailers to our list or we might consider relaxing this.
RESERVED_TRAILER_PREFIXES = ["ci-", "bewnchmark-", "skip-"]
ALL_KEY = "all"

# Note that these are fnmatch patterns, which are not the same as gitignore
# patterns because they don't treat '/' specially. The standard library doesn't
# contain a function for gitignore style "wildmatch". There's a third-party
# library pathspec (https://pypi.org/project/pathspec/), but it doesn't seem
# worth the dependency.
SKIP_PATH_PATTERNS = [
    "docs/*",
    "third_party/mkdocs-material/*",
    "experimental/*",
    # These configure the runners themselves and don't affect presubmit.
    "build_tools/github_actions/runner/*",
    ".github/ISSUE_TEMPLATE/*",
    "*.cff",
    "*.clang-format",
    "*.gitignore",
    "*.git-blame-ignore-revs",
    "*.md",
    "*.natvis",
    "*.pylintrc",
    "*.rst",
    "*.toml",
    "*.yamllint.yml",
    "*.yapf",
    "*CODEOWNERS",
    "*AUTHORS",
    "*LICENSE",
]

RUNNER_ENV_DEFAULT = "prod"
RUNNER_ENV_OPTIONS = [RUNNER_ENV_DEFAULT, "testing"]

CONTROL_JOBS = frozenset(["setup", "summary"])

# Jobs to run only on postsubmit by default.
# They may also run on presubmit only under certain conditions.
DEFAULT_POSTSUBMIT_ONLY_JOBS = frozenset(
    [
        "build_test_all_arm64",
        "build_test_all_windows",
        "build_test_all_macos_arm64",
        "build_test_all_macos_x86_64",
        "debug_windows",
        # Due to the outstock of A100, only run this test in postsubmit.
        "test_a100",
    ]
)

# Jobs to run in presumbit if files under the corresponding path see changes.
# Each tuple consists of the CI job name and a list of file paths to match.
# The file paths should be specified using Unix shell-style wildcards.
PRESUBMIT_TOUCH_ONLY_JOBS = [
    ("build_test_all_macos_arm64", ["runtime/src/iree/hal/drivers/metal/*"]),
    ("build_test_all_windows", ["*win32*", "*windows*", "*msvc*"]),
    ("debug_windows", ["*win32*", "*windows*", "*msvc*"]),
]

# Default presets enabled in CI.
DEFAULT_BENCHMARK_PRESET_GROUP = [
    preset
    for preset in benchmark_presets.DEFAULT_PRESETS
    # RISC-V benchmarks haven't been supported in CI workflow.
    if preset not in [benchmark_presets.RISCV]
] + ["comp-stats"]
DEFAULT_BENCHMARK_PRESET = "default"
LARGE_BENCHMARK_PRESET_GROUP = benchmark_presets.LARGE_PRESETS
# All available benchmark preset options including experimental presets.
BENCHMARK_PRESET_OPTIONS = (
    benchmark_presets.ALL_EXECUTION_PRESETS + benchmark_presets.ALL_COMPILATION_PRESETS
)
BENCHMARK_LABEL_PREFIX = "benchmarks"

PR_DESCRIPTION_TEMPLATE = string.Template("${title}\n\n${body}")

# Patterns to detect "LLVM integration" PRs, i.e. changes that update the
# third_party/llvm-project submodule. This should only include PRs
# intended to be merged and should exclude test/draft PRs as well as
# PRs that include temporary patches to the submodule during review.
# See also: https://github.com/openxla/iree/issues/12268
LLVM_INTEGRATE_TITLE_PATTERN = re.compile("^integrate.+llvm", re.IGNORECASE)
LLVM_INTEGRATE_BRANCH_PATTERN = re.compile(
    "bump-llvm|llvm-bump|integrate-llvm", re.IGNORECASE
)
LLVM_INTEGRATE_LABEL = "llvm-integrate"


def skip_path(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in SKIP_PATH_PATTERNS)


def set_output(d: Mapping[str, str]):
    print(f"Setting outputs: {d}")
    step_output_file = os.environ["GITHUB_OUTPUT"]
    with open(step_output_file, "a") as f:
        f.writelines(f"{k}={v}" + "\n" for k, v in d.items())


def write_job_summary(summary: str):
    """Write markdown messages on Github workflow UI.
    See https://docs.github.com/en/actions/using-workflows/workflow-commands-for-github-actions#adding-a-job-summary
    """
    step_summary_file = os.environ["GITHUB_STEP_SUMMARY"]
    with open(step_summary_file, "a") as f:
        # Use double newlines to split sections in markdown.
        f.write(summary + "\n\n")


def check_description_and_show_diff(
    original_description: str,
    original_labels: Sequence[str],
    current_description: str,
    current_labels: Sequence[str],
):
    original_labels = sorted(original_labels)
    current_labels = sorted(current_labels)
    if (
        original_description == current_description
        and original_labels == current_labels
    ):
        return

    description_diffs = difflib.unified_diff(
        original_description.splitlines(keepends=True),
        current_description.splitlines(keepends=True),
    )
    description_diffs = "".join(description_diffs)

    if description_diffs != "":
        description_diffs = textwrap.dedent(
            """\
    ```diff
    {}
    ```
    """
        ).format(description_diffs)

    if original_labels == current_labels:
        label_diffs = ""
    else:
        label_diffs = textwrap.dedent(
            """\
    ```
    Original labels: {original_labels}
    Current labels: {current_labels}
    ```
    """
        ).format(original_labels=original_labels, current_labels=current_labels)

    write_job_summary(
        textwrap.dedent(
            """\
  :pushpin: Using the PR description and labels different from the original PR event that started this workflow.

  <details>
  <summary>Click to show diff (original vs. current)</summary>

  {description_diffs}

  {label_diffs}
  </details>"""
        ).format(description_diffs=description_diffs, label_diffs=label_diffs)
    )


def get_trailers_and_labels(is_pr: bool) -> Tuple[Mapping[str, str], List[str]]:
    if not is_pr:
        return ({}, [])

    title = os.environ["PR_TITLE"]
    body = os.environ.get("PR_BODY", "")
    labels = json.loads(os.environ.get("PR_LABELS", "[]"))
    original_title = os.environ.get("ORIGINAL_PR_TITLE")
    original_body = os.environ.get("ORIGINAL_PR_BODY", "")
    original_labels = json.loads(os.environ.get("ORIGINAL_PR_LABELS", "[]"))

    description = PR_DESCRIPTION_TEMPLATE.substitute(title=title, body=body)

    # PR information can be fetched from API for the latest updates. If
    # ORIGINAL_PR_TITLE is set, compare the current and original description and
    # show a notice if they are different. This is mostly to inform users that
    # the workflow might not parse the PR description they expect.
    if original_title is not None:
        original_description = PR_DESCRIPTION_TEMPLATE.substitute(
            title=original_title, body=original_body
        )
        print(
            "Original PR description and labels:",
            original_description,
            original_labels,
            sep="\n",
        )
        check_description_and_show_diff(
            original_description=original_description,
            original_labels=original_labels,
            current_description=description,
            current_labels=labels,
        )

    print("Parsing PR description and labels:", description, labels, sep="\n")

    trailer_lines = subprocess.run(
        ["git", "interpret-trailers", "--parse", "--no-divider"],
        input=description,
        stdout=subprocess.PIPE,
        check=True,
        text=True,
        timeout=60,
    ).stdout.splitlines()
    trailer_map = {
        k.lower().strip(): v.strip()
        for k, v in (line.split(":", maxsplit=1) for line in trailer_lines)
    }

    for key in trailer_map:
        if not Trailer.contains(key):
            for prefix in RESERVED_TRAILER_PREFIXES:
                if key.startswith(prefix):
                    print(
                        f"Trailer '{key}' starts with reserved prefix"
                        f"'{prefix}' but is unknown."
                    )
            print(f"Skipping unknown trailer '{key}'", file=sys.stderr)

    return (trailer_map, labels)


def get_modified_paths(base_ref: str) -> Optional[Iterable[str]]:
    """Returns the paths of modified files in this code change."""
    try:
        return subprocess.run(
            ["git", "diff", "--name-only", base_ref],
            stdout=subprocess.PIPE,
            check=True,
            text=True,
            timeout=60,
        ).stdout.splitlines()
    except TimeoutError as e:
        print(
            "Computing modified files timed out. Not using PR diff to determine"
            " jobs to run.",
            file=sys.stderr,
        )
        return None


def modifies_non_skip_paths(paths: Optional[Iterable[str]]) -> bool:
    """Returns true if not all modified paths are in the skip set."""
    if paths is None:
        return True
    return any(not skip_path(p) for p in paths)


def get_runner_env(trailers: Mapping[str, str]) -> str:
    runner_env = trailers.get(Trailer.RUNNER_ENV)
    if runner_env is None:
        print(
            f"Using '{RUNNER_ENV_DEFAULT}' runners because"
            f" '{Trailer.RUNNER_ENV}' not found in {trailers}"
        )
        runner_env = RUNNER_ENV_DEFAULT
    else:
        print(f"Using runner environment '{runner_env}' from PR description trailers")
    return runner_env


def parse_jobs_trailer(
    trailers: Mapping[str, str], key: str, all_jobs: Set[str]
) -> Set[str]:
    jobs_text = trailers.get(key)
    if jobs_text is None:
        return set()
    jobs = set(name.strip() for name in jobs_text.split(","))
    if ALL_KEY in jobs:
        if len(jobs) != 1:
            raise ValueError(
                f"'{ALL_KEY}' must be alone in job specification"
                f" trailer, but got '{key}: {jobs_text}'"
            )
        print(f"Expanded trailer '{key}: {jobs_text}' to all jobs")
        return all_jobs

    jobs = set(jobs)
    unknown_jobs = jobs - all_jobs
    if unknown_jobs:
        # Unknown jobs may be for a different workflow. Warn then continue.
        print(f"::warning::Unknown jobs '{','.join(unknown_jobs)}' in trailer '{key}'")
        jobs = jobs - unknown_jobs
    return jobs


def parse_path_from_workflow_ref(repo: str, workflow_ref: str) -> pathlib.Path:
    if not workflow_ref.startswith(repo):
        raise ValueError(
            "Can't parse the external workflow ref"
            f" '{workflow_ref}' outside the repo '{repo}'."
        )
    # The format of workflow ref: `${repo}/${workflow file path}@${ref}`
    workflow_file = workflow_ref[len(repo) :].lstrip("/")
    workflow_file = workflow_file.split("@", maxsplit=1)[0]
    return pathlib.Path(workflow_file)


def parse_jobs_from_workflow_file(workflow_file: pathlib.Path) -> Set[str]:
    print(f"Parsing workflow file: '{workflow_file}'.")

    workflow = yaml.load(workflow_file.read_text(), Loader=yaml.SafeLoader)
    all_jobs = set(workflow["jobs"].keys())
    all_jobs -= CONTROL_JOBS

    if ALL_KEY in all_jobs:
        raise ValueError(f"Workflow has job with reserved name '{ALL_KEY}'")
    return all_jobs


def get_enabled_jobs(
    trailers: Mapping[str, str],
    all_jobs: Set[str],
    *,
    is_pr: bool,
    modified_paths: Optional[Iterable[str]],
) -> Set[str]:
    """Returns the CI jobs to run.

    Args:
      trailers: trailers from PR description.
      all_jobs: all known supported jobs.
      is_pr: whether this is for pull requests or not.
      modified_paths: the paths of the files changed. These paths are
        relative to the repo root directory.

    Returns:
      The list of CI jobs to run.
    """
    if not is_pr:
        print(
            "Running all jobs because run was not triggered by a pull request"
            " event.",
            file=sys.stderr,
        )
        return all_jobs

    if Trailer.SKIP_CI in trailers:
        if (
            Trailer.EXACTLY_JOBS in trailers
            or Trailer.EXTRA_JOBS in trailers
            or Trailer.SKIP_JOBS in trailers
        ):
            raise ValueError(
                f"Cannot specify both '{Trailer.SKIP_JOBS}' and any of"
                f" '{Trailer.EXACTLY_JOBS}', '{Trailer.EXTRA_JOBS}',"
                f" '{Trailer.SKIP_JOBS}'"
            )
        print(
            f"Skipping all jobs because PR description has"
            f" '{Trailer.SKIP_CI}' trailer."
        )
        return set()

    if Trailer.EXACTLY_JOBS in trailers:
        if Trailer.EXTRA_JOBS in trailers or Trailer.SKIP_JOBS in trailers:
            raise ValueError(
                f"Cannot mix trailer '{Trailer.EXACTLY_JOBS}' with"
                f" '{Trailer.EXTRA_JOBS}' or '{Trailer.SKIP_JOBS}'"
            )

        exactly_jobs = parse_jobs_trailer(
            trailers,
            Trailer.EXACTLY_JOBS,
            all_jobs,
        )
        return exactly_jobs

    skip_jobs = parse_jobs_trailer(trailers, Trailer.SKIP_JOBS, all_jobs)
    extra_jobs = parse_jobs_trailer(trailers, Trailer.EXTRA_JOBS, all_jobs)

    ambiguous_jobs = skip_jobs & extra_jobs
    if ambiguous_jobs:
        raise ValueError(
            f"Jobs cannot be specified in both '{Trailer.SKIP_JOBS}' and"
            f" '{Trailer.EXTRA_JOBS}', but found {ambiguous_jobs}"
        )

    default_jobs = all_jobs - DEFAULT_POSTSUBMIT_ONLY_JOBS

    if not modifies_non_skip_paths(modified_paths):
        print(
            "Not including any jobs by default because all modified files"
            " are marked as excluded."
        )
        default_jobs = frozenset()
    else:
        # Add jobs if the monitored files are changed.
        for modified_path in modified_paths:
            for job, match_paths in PRESUBMIT_TOUCH_ONLY_JOBS:
                for match_path in match_paths:
                    if fnmatch.fnmatch(modified_path, match_path):
                        default_jobs |= {job}

    return (default_jobs | extra_jobs) - skip_jobs


def get_benchmark_presets(
    trailers: Mapping[str, str],
    labels: Sequence[str],
    is_pr: bool,
    is_llvm_integrate_pr: bool,
) -> str:
    """Parses and validates the benchmark presets from trailers.

    Args:
      trailers: trailers from PR description.
      labels: list of PR labels.
      is_pr: is pull request event.
      is_llvm_integrate_pr: is LLVM integration PR.

    Returns:
      A comma separated preset string, which later will be parsed by
      build_tools/benchmarks/export_benchmark_config.py.
    """

    skip_llvm_integrate_benchmark = Trailer.SKIP_LLVM_INTEGRATE_BENCHMARK in trailers
    if skip_llvm_integrate_benchmark:
        print(
            f"Skipping default benchmarking on LLVM integration because PR "
            f"description has '{Trailer.SKIP_LLVM_INTEGRATE_BENCHMARK}'"
            f" trailer."
        )

    if not is_pr:
        preset_options = {DEFAULT_BENCHMARK_PRESET}
        print(f"Using benchmark presets '{preset_options}' for non-PR run")
    elif is_llvm_integrate_pr and not skip_llvm_integrate_benchmark:
        # Run all benchmark presets for LLVM integration PRs.
        preset_options = {DEFAULT_BENCHMARK_PRESET}
        print(f"Using benchmark preset '{preset_options}' for LLVM integration PR")
    else:
        preset_options = set(
            label.split(":", maxsplit=1)[1]
            for label in labels
            if label.startswith(BENCHMARK_LABEL_PREFIX + ":")
        )
        trailer = trailers.get(Trailer.BENCHMARK_EXTRA)
        if trailer is not None:
            preset_options = preset_options.union(
                option.strip() for option in trailer.split(",")
            )
            print(
                f"Using benchmark preset '{preset_options}' from trailers"
                f" and labels"
            )

    if DEFAULT_BENCHMARK_PRESET in preset_options:
        preset_options.remove(DEFAULT_BENCHMARK_PRESET)
        preset_options.update(DEFAULT_BENCHMARK_PRESET_GROUP)

    if preset_options.intersection(DEFAULT_BENCHMARK_PRESET_GROUP):
        # The is a sugar to run the compilation benchmarks when any default
        # benchmark preset is present.
        preset_options.add("comp-stats")

    preset_options = sorted(preset_options)
    for preset_option in preset_options:
        if preset_option not in BENCHMARK_PRESET_OPTIONS:
            raise ValueError(
                f"Unknown benchmark preset option: '{preset_option}'.\n"
                f"Available options: '{BENCHMARK_PRESET_OPTIONS}'."
            )

    return ",".join(preset_options)


def main():
    is_pr = os.environ["GITHUB_EVENT_NAME"] == "pull_request"
    trailers, labels = get_trailers_and_labels(is_pr)
    is_llvm_integrate_pr = bool(
        LLVM_INTEGRATE_TITLE_PATTERN.search(os.environ.get("PR_TITLE", ""))
        or LLVM_INTEGRATE_BRANCH_PATTERN.search(os.environ.get("PR_BRANCH", ""))
        or LLVM_INTEGRATE_LABEL in labels
    )
    repo = os.environ["GITHUB_REPOSITORY"]
    workflow_ref = os.environ["GITHUB_WORKFLOW_REF"]
    workflow_file = parse_path_from_workflow_ref(repo=repo, workflow_ref=workflow_ref)
    base_ref = os.environ["BASE_REF"]

    try:
        benchmark_presets = get_benchmark_presets(
            trailers, labels, is_pr, is_llvm_integrate_pr
        )
        all_jobs = parse_jobs_from_workflow_file(workflow_file)
        enabled_jobs = get_enabled_jobs(
            trailers,
            all_jobs,
            modified_paths=get_modified_paths(base_ref),
            is_pr=is_pr,
        )
    except ValueError as e:
        print(e)
        sys.exit(1)
    output = {
        "enabled-jobs": json.dumps(sorted(enabled_jobs)),
        "is-pr": json.dumps(is_pr),
        "runner-env": get_runner_env(trailers),
        "runner-group": "presubmit" if is_pr else "postsubmit",
        "write-caches": "0" if is_pr else "1",
        "benchmark-presets": benchmark_presets,
    }

    set_output(output)


if __name__ == "__main__":
    main()
