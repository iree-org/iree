# Contributing

This is a more detailed version of the top-level
[CONTRIBUTING.md](/CONTRIBUTING.md) file. We keep it separate to avoid everyone
getting a pop-up when creating a PR after each time it changes.

## Downstream Integrations

Due to the process by which we synchronize this GitHub project with our internal
Google source code repository, there are some oddities in our workflows and
processes. We aim to minimize these, and especially to mitigate their impact on
external contributors, but when they come up, they are documented here for
clarity and transparency. If any of these things are particularly troublesome or
painful for your workflow, please reach out to us so we can prioritize a fix.

Hopefully these quirks actually make usage in other downstream projects easier,
but integrators may need to look past some details (like the Bazel build system,
Android support, etc.) based on their specific requirements.

## Build Systems

IREE supports building from source with both Bazel and CMake. CMake is the
preferred build system for open source users and offers the most flexible
configuration options. Bazel is a stricter build system and helps with usage in
the Google internal source repository. Certain dependencies (think large/complex
projects like CUDA, TensorFlow, PyTorch, etc.) may be difficult to support with
one build system or the other, so the project may configure these as optional.

## CI

IREE uses [GitHub Actions](https://docs.github.com/en/actions) for CI. The
primary CI is configured in the
[ci.yml workflow file](/.github/workflows/ci.yml).

### Self-Hosted Runners

In addition to the default runners GitHub provides, IREE uses
[self-hosted runners](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners)
to run many of its workflow jobs. These enable access to additional compute and
custom configurations such as accelarators. Configuration scripting is checked
in to this repository (see the
[README for that directory](/build_tools/github_actions/runner/README.md)).

### Custom Managed Runners

In addition to our self-hosted runners, we use GitHub's
[large managed runners](https://docs.github.com/en/actions/using-github-hosted-runners/about-larger-runners)
for some platforms that are more trouble to configure ourselves (e.g. Mac).

### CI Behavior Manipulation

The setup step of the CI determines which CI jobs to run. This is controlled by
the [configure_ci.py](/build_tools/github_actions/configure_ci.py) script. It
will generally run a pre-determined set of jobs on presubmit with some jobs kept
as post-submit only. If changes are only to a certain set of excluded files that
we know don't affect CI (e.g. docs), then it will skip the jobs. You can
customize which jobs run using
[git trailers](https://git-scm.com/docs/git-interpret-trailers) in the PR
description. The available options are

``` text
ci-skip: jobs,to,skip
ci-extra: extra,jobs,to,run
ci-exactly: exact,set,of,jobs,to,run
skip-ci: free form reason
skip-llvm-integrate-benchmark: free form reason
benchmark-extra: extra,benchmarks,to,run
runner-env: [testing|prod]
```

The first three follow the same format and instruct the setup script on which
jobs to include or exclude from its run. They take a comma-separated list of
jobs which must be from the set of top-level job identifiers in ci.yml file or
the special keyword "all" to indicate all jobs. `ci-skip` removes jobs that
would otherwise be included, though it is not an error to list jobs that would
not be included by default. `ci-extra` adds additional jobs that would not have
otherwise been run, though it is not an error to list jobs that would have been
included anyway. It *is* an error to list a job in both of these fields.
`ci-exactly` provides an exact list of jobs that should run. It is mutually
exclusive with both `ci-skip` and `ci-extra`. In all these cases, the setup does
not make any effort to ensure that job dependencies are satisfied. Thus, if you
request skipping the `build_all` job, all the jobs that depend on it will fail,
not be skipped. `skip-ci` is an older option that simply skips all jobs. Its
usage is deprecated and it is mutually exclusive with all of the other `ci-*`
options. Prefer `ci-skip: all`.

Benchmarks don't run by default on PRs, and must be specifically requested. They
*do* run by default on PRs detected to be an integration of LLVM into IREE, but
this behavior can be disabled with `skip-llvm-integrate-benchmark`. The
`benchmark-extra` option allows specifying additional benchmark presets to run
as part of benchmarking. It accepts a comma-separated list of benchmark presets.
This combines with labels added to the PR (which are a more limited set of
options). See the [benchmark suites documentation](./benchmark_suites.md).

The `runner-env` option controls which runner environment to target for our
self-hosted runners. We maintain a test environment to allow testing out new
configurations prior to rolling them out. This trailer is for advanced users who
are working on the CI infrastructure itself.

#### CI configuration recipes

Copy/paste any of these at the bottom of a PR description to change what the CI
runs.

* Also run Windows and macOS builds that are normally post-merge only:

  ``` text
  ci-extra: build_test_all_windows,build_test_all_macos_arm64,build_test_all_macos_x86_64
  ```

* Also run GPU tests on NVIDIA A100 runners (opt-in due to low availability):

  ``` text
  ci-extra: test_a100
  ```

* Skip all CI builds and tests, e.g. for comment-only changes:

  ``` text
  skip-ci: Comment-only change.
  ```

* Only run Bazel builds, e.g. for changes only affecting Bazel rules:

  ``` text
  ci-exactly: build_test_all_bazel
  ```

For example, this PR opted in to running the `build_test_all_windows` job:

![ci-extra](/docs/developers/assets/ci-extra.png)

The enabled jobs can be viewed from the Summary page of an action run:

![ci_enabled_jobs](/docs/developers/assets/ci_enabled_jobs.png)
