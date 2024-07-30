---
icon: octicons/code-review-16
---

# Contributing to IREE

We'd love to accept your patches and contributions to this project.

!!! note "Note - coordinating efforts"

    Please [file issues](https://github.com/iree-org/iree/issues/new/choose) or
    reach out on any of our other
    [communication channels](../../index.md#communication-channels) before doing
    substantial work; this will ensure that others don't duplicate the work and
    that there's a chance to discuss any design issues.

## Developer policies

### :octicons-code-of-conduct-16: Code of conduct

This project follows the
[LF Projects code of conduct](https://lfprojects.org/policies/code-of-conduct/).

### :octicons-law-16: Developer Certificate of Origin

Contributors must certify that they wrote or otherwise have the right to submit
the code they are contributing to the project.

??? quote "Expand to read the full DCO agreement text"

    By making a contribution to this project, I certify that:

    1. The contribution was created in whole or in part by me and I have the
      right to submit it under the open source license indicated in the file; or

    2. The contribution is based upon previous work that, to the best of my
      knowledge, is covered under an appropriate open source license and I have
      the right under that license to submit that work with modifications, whether
      created in whole or in part by me, under the same open source license
      (unless I am permitted to submit under a different license), as indicated
      in the file; or

    3. The contribution was provided directly to me by some other person who
      certified 1., 2. or 3. and I have not modified it.

    4. I understand and agree that this project and the contribution are public
      and that a record of the contribution (including all personal information
      I submit with it, including my sign-off) is maintained indefinitely and
      may be redistributed consistent with this project or the open source
      license(s) involved.

Signing is enforced by the [DCO GitHub App](https://github.com/apps/dco) (see
also the [dcoapp/app](https://github.com/dcoapp/app) repository).

The DCO check requires that all commits included in pull requests _either_
are cryptographically signed by a member of the repository's organization _or_
include a `Signed-off-by` message as a git trailer.

#### Crypographically signing commits

_This is the recommended approach for frequent contributors!_

For members of the repository's organization
(see [obtaining commit access](#obtaining-commit-access)), commits that are
signed do not require the `Signed-off-by` text. See these references:

* [Signing commits](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits)
    (generate key, add to <https://github.com/settings/keys>, `git commit -S`)
* [SSH commit signature verification](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification#ssh-commit-signature-verification)
    (recommended if you already use SSH keys with GitHub) and
    [Signing Git Commits with SSH Keys](https://blog.dbrgn.ch/2021/11/16/git-ssh-signatures/)
    (streamlined version of the previous page).

    SSH keys can be added at <https://github.com/settings/ssh/new>
    (Note that even if you have added your SSH key as an authorized key, you
    need to add it again as a signing key).

    Then,

    ```bash
    # Sign commits automatically
    git config --global commit.gpgsign true
    git config --global tag.gpgsign true

    # Sign using SSH, not GPG
    git config --global user.signingkey ~/.ssh/id_rsa.pub
    git config --global gpg.format ssh

    # Create an "allowed_signers" file
    echo your@email `cat ~/.ssh/id_rsa.pub` > ~/.ssh/allowed_signers
    git config --global gpg.ssh.allowedSignersFile ~/.ssh/allowed_signers
    ```

* [Generating GPG keys](https://docs.github.com/en/authentication/managing-commit-signature-verification/generating-a-new-gpg-key)
    (alternative to using SSH keys)

    GPG keys can be added at <https://github.com/settings/gpg/new>, then:

    ```bash
    # Sign commits automatically
    git config --global commit.gpgsign true
    git config --global tag.gpgsign true
    ```

#### Adding `Signed-off-by` to commits

_This requires less setup and is suitable for first time contributors._

Contributors _sign-off_ their agreement by adding a `Signed-off-by` line to
commit messages:

```text
This is my commit message

Signed-off-by: Random J Developer <random@developer.example.org>
```

* Git will automatically append this message if you use the `-s` option:

    ```bash
    git commit -s -m 'This is my commit message'
    ```

* Users of [Visual Studio Code](https://code.visualstudio.com/) can add
  `"git.alwaysSignOff": true,` in their settings

* See `.git/hooks/prepare-commit-msg.sample` for how to automatically
  add this using a [git hook](https://git-scm.com/docs/githooks)

### :octicons-people-16: AUTHORS, CODEOWNERS, and MAINTAINERS

The [`AUTHORS` file](https://github.com/iree-org/iree/blob/main/AUTHORS) keeps
track of those who have made significant contributions to the project.

* If you would like additional recognition for your contributions, you may add
  yourself or your organization (please add the entity who owns the copyright
  for your contributions).
* The source control history remains the most accurate source for individual
  contributions.

The
[`.github/CODEOWNERS` file](https://github.com/iree-org/iree/blob/main/.github/CODEOWNERS)
lets maintainers opt in to PR reviews modifying certain paths.

* Review is not required from a code owner, though it is recommended.

The
[`MAINTAINERS.md` file](https://github.com/iree-org/iree/blob/main/MAINTAINERS.md)
documents official maintainers for project components.

## :octicons-code-16: Coding policies

### :octicons-pencil-16: Coding style guidelines

Most of the code style is derived from the
[Google Style Guides](http://google.github.io/styleguide/) for the appropriate
language. The C++ compiler portion of the project follows the
[MLIR/LLVM style guide](https://mlir.llvm.org/getting_started/DeveloperGuide/#style-guide).

We use [pre-commit](https://pre-commit.com/) to run assorted formatters and lint
checks. The configuration file at
[`.pre-commit-config.yaml`](https://github.com/iree-org/iree/blob/main/.pre-commit-config.yaml)
defines which "hooks" run.

* To run these hooks on your local commits, follow the
  [pre-commit installation instructions](https://pre-commit.com/#installation).
* Individual formatters like
  [`clang-format`](https://clang.llvm.org/docs/ClangFormat.html) (C/C++) and
  [_Black_](https://black.readthedocs.io/en/stable/) (Python) can also be set to
  run automatically in your editor of choice.

!!! note

    Improvements to code structure and clarity are welcome but please file
    issues to track such work first. Pure style changes are unlikely to be
    accepted unless they are applied consistently across the project.

### :material-test-tube: Testing policy

With few exceptions, features should be accompanied by automated tests.

We use a mix of in-tree and out-of-tree unit and integration tests. For more
information about the types of tests used across the project, refer to the
[testing guide](./testing-guide.md).

## :simple-github: GitHub policies

### :octicons-code-review-16: Code reviews

All submissions, including submissions by maintainers, require review. We
use GitHub pull requests (PRs) for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

* Please keep PRs small (focused on a single issue) to make reviews and later
  culprit-finding easier.
* You may see trusted core contributors bending this rule for project
  maintenance and major subsystem renovation. If you feel like the rules aren't
  working for a certain situation, please ask as we bias towards pragmatism for
  cases that require it.

### :material-check-all: GitHub Actions workflows

We use [GitHub Actions](https://docs.github.com/en/actions) to automatically
build and test various parts of the project.

* Most presubmit workflows will only run automatically on PRs if you are a
  project collaborator. Otherwise a maintainer must
  [approve workflow runs](https://docs.github.com/en/actions/managing-workflow-runs/approving-workflow-runs-from-public-forks).
  If you are sending code changes to the project, please
  [request commit access](#obtaining-commit-access), so that these can run
  automatically.
* It is generally expected that PRs will only be merged when all checks are
  passing. In some cases, pre-existing failures may be bypassed by a maintainer.

??? tip - "Tip - adjusting workflow behavior"

    Some workflows only run on commits after they are merged. See the
    [CI behavior manipulation](#ci-behavior-manipulation) section below to
    learn how to customize this behavior.

### :octicons-git-pull-request-16: Merging approved changes

After review and presubmit checks, PRs should typically be merged using
"squash and merge".

* The squashed commit summary should match the PR title and the commit
  description should match the PR body (this is the default behavior).
  Accordingly, please write these as you would a helpful commit message.

It is assumed that the PR author will merge their change unless they ask
someone else to merge it for them (e.g. because they don't have write access
yet).

### :octicons-git-merge-16: Obtaining commit access

Access to affiliated repositories is divided into tiers:

| Tier | Description | Team link |
| ---- | ----------- | --------- |
Triage | **New project members should typically start here**<br>:material-check: Can be [assigned issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/assigning-issues-and-pull-requests-to-other-github-users)<br>:material-check: Can apply labels to issues / PRs<br>:material-check: Can run workflows [without approval](https://docs.github.com/en/actions/managing-workflow-runs/approving-workflow-runs-from-public-forks) | [iree-triage](https://github.com/orgs/iree-org/teams/iree-triage)
Write | **Established project contributors should request this access**<br>:material-check: Can [merge approved pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/merging-a-pull-request)<br>:material-check: Can create branches | [iree-write](https://github.com/orgs/iree-org/teams/iree-write)
Maintain/Admin | :material-check: Can [edit repository settings](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features)<br>:material-check: Can push to [protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches) | Added case-by-case

All access tiers first require joining the
[iree-org GitHub organization](https://github.com/iree-org/).

<!-- markdownlint-disable-next-line -->
[Fill out this form to request access :fontawesome-solid-paper-plane:](https://docs.google.com/forms/d/e/1FAIpQLSfEwANtMvLJWq-ED4lub_xsMch0MgNY02VxgtXE61FqNvNVUg/viewform){ .md-button .md-button--primary }

Once you are a member of the iree-org GitHub organization, you can request to
join any of the teams on <https://github.com/orgs/iree-org/teams>.

### :octicons-git-branch-16: Branch naming

Most work should be done on
[repository forks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks).
For developers with write access, when creating a branch in the common
[iree-org/iree repository](https://github.com/iree-org/iree), please follow
these naming guidelines:

Branch type | Naming scheme | Example
-- | -- | --
Single user | `users/[username]/*` | `users/cooldeveloper/my-awesome-feature`
Shared feature branch | `shared/*` | `shared/pytorch-performance-sprint`
Dependency updates | `integrates/*` | `integrates/llvm-20240501`

Branches that do not meet these guidelines may be deleted, especially if
they [appear to be stale](https://github.com/iree-org/iree/branches/stale).

## Tips for contributors

### Tool recommendations

| Program or tool | Description |
| -- | -- |
[:material-microsoft-visual-studio-code: Visual Studio Code (VSCode)](<https://code.visualstudio.com/>) | The most commonly used editor amongst IREE developers
[:simple-cmake: Ccache](<https://ccache.dev/>) | A fast C/C++ compiler cache. See the [CMake with `ccache`](../building/cmake-with-ccache.md) page
[:simple-github: GitHub CLI](<https://github.com/cli/cli>) | A CLI for interacting with GitHub
[:simple-github: "Refined GitHub" browser extensions](<https://github.com/sindresorhus/refined-github>) | Extension that add features to the GitHub UI

### :material-hammer-wrench: Build systems

IREE supports building from source with both Bazel and CMake.

* CMake is the preferred build system and offers the most flexible
  configuration options
* Bazel is a stricter build system and helps with usage in Google's downstream
  source repository
* Certain dependencies (think large/complex projects like CUDA, TensorFlow,
  PyTorch, etc.) may be difficult to support with one build system or the
  other, so the project may configure these as optional

### :octicons-server-16: Continuous integration (CI)

IREE uses [GitHub Actions](https://docs.github.com/en/actions) for CI. The
primary CI is configured in the
[ci.yml workflow file](https://github.com/iree-org/iree/blob/main/.github/workflows/ci.yml).

#### Self-hosted runners

In addition to the default runners GitHub provides, IREE uses
[self-hosted runners](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners)
to run many of its workflow jobs. These enable access to additional compute and
custom configurations such as accelerators.

* Configuration for GCP runners is stored at
[`build_tools/github_actions/runner/`](https://github.com/iree-org/iree/blob/main/build_tools/github_actions/runner/)
* Configuration for other runners is done manually as needed

#### CI behavior manipulation

The setup step of the CI determines which CI jobs to run. This is controlled by
the
[configure_ci.py](https://github.com/iree-org/iree/blob/main/build_tools/github_actions/configure_ci.py)
script. It will generally run a pre-determined set of jobs on presubmit with
some jobs kept as post-submit only. If changes are only to a certain set of
excluded files that we know don't affect CI (e.g. Markdown files), then it will
skip the jobs.

You can customize which jobs run using
[git trailers](https://git-scm.com/docs/git-interpret-trailers) in the PR
description.

The available options are

``` text
ci-skip: jobs,to,skip
ci-extra: extra,jobs,to,run
ci-exactly: exact,set,of,jobs,to,run
skip-ci: free form reason
skip-llvm-integrate-benchmark: free form reason
benchmark-extra: extra,benchmarks,to,run
runner-env: [testing|prod]
```

??? info - "Using `skip-ci`"

    `skip-ci` skips all jobs. It is mutually exclusive with the other `ci-*`
    options and is synonomous with `ci-skip: all`.

    ``` text
    skip-ci: free form reason
    ```

??? info - "Using `ci-skip`, `ci-extra`, `ci-exactly`"

    The `ci-*` options instruct the setup script on which jobs to include or
    exclude from its run. They take a comma-separated list of jobs which must be
    from the set of top-level job identifiers in the `ci.yml` file or the
    special keyword "all" to indicate all jobs.

    ``` text
    ci-skip: jobs,to,skip
    ci-extra: extra,jobs,to,run
    ci-exactly: exact,set,of,jobs,to,run
    ```

    * `ci-skip` removes jobs that would otherwise be included, though it is not
    an error to list jobs that would not be included by default.
    * `ci-extra` adds additional jobs that would not have otherwise been run,
    though it is not an error to list jobs that would have been included anyway.
    It *is* an error to list a job in both "skip" and "extra".
    * `ci-exactly` provides an exact list of jobs that should run. It is
    mutually exclusive with both "skip" and "extra".

    In all these cases, the setup does not make any effort to ensure that job
    dependencies are satisfied. Thus, if you request skipping the `build_all`
    job, all the jobs that depend on it will fail, not be skipped.

??? info - "Using `benchmark-extra`, `skip-llvm-integrate-benchmark`"

    ``` text
    benchmark-extra: extra,benchmarks,to,run
    skip-llvm-integrate-benchmark: free form reason
    ```

    Benchmarks don't run by default on PRs, and must be specifically requested.

    The `benchmark-extra` option allows specifying additional benchmark presets
    to run as part of benchmarking. It accepts a comma-separated list of
    benchmark presets. This combines with labels added to the PR (which are a
    more limited set of options). See the
    [benchmark suites documentation](../performance/benchmark-suites.md).

    Benchmarks *do* run by default on PRs detected to be an integration of LLVM
    into IREE, but this behavior can be disabled with
    `skip-llvm-integrate-benchmark`.

??? info - "Using `runner-env`"

    The `runner-env` option controls which runner environment to target for our
    self-hosted runners. We maintain a test environment to allow testing out new
    configurations prior to rolling them out. This trailer is for advanced users
    who are working on the CI infrastructure itself.

    ``` text
    runner-env: [testing|prod]
    ```

##### CI configuration recipes

Copy/paste any of these at the bottom of a PR description to change what the CI
runs.

* Also run Windows and macOS builds that are normally post-merge only:

    ``` text
    ci-extra: build_test_all_windows,build_test_all_macos_arm64,build_test_all_macos_x86_64
    ```

* Also run GPU tests on NVIDIA A100 runners (opt-in due to low availability):

    ``` text
    ci-extra: test_nvidia_a100
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

![ci-extra](./contributing-ci-extra.png)

The enabled jobs can be viewed from the Summary page of an action run:

![ci_enabled_jobs](./contributing-ci-enabled-jobs.png)
