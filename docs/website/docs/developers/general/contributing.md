# Contributing

This is a more detailed version of the top-level
[CONTRIBUTING.md](https://github.com/openxla/iree/blob/main/CONTRIBUTING.md)
file. We keep it separate to avoid everyone getting a pop-up when creating a PR
after each time it changes.

<!-- TODO(scotttodd): Update this document
    * pull more text into this, update that to point to the website
    * document access controls (join organization then team)
    * document revert policy
    * document where new community members should start
-->

## Build systems

IREE supports building from source with both Bazel and CMake.

* CMake is the preferred build system and offers the most flexible
  configuration options
* Bazel is a stricter build system and helps with usage in Google's downstream
  source repository
* Certain dependencies (think large/complex projects like CUDA, TensorFlow,
  PyTorch, etc.) may be difficult to support with one build system or the
  other, so the project may configure these as optional

## Continuous integration (CI)

IREE uses [GitHub Actions](https://docs.github.com/en/actions) for CI. The
primary CI is configured in the
[ci.yml workflow file](https://github.com/openxla/iree/blob/main/.github/workflows/ci.yml).

### Self-hosted runners

In addition to the default runners GitHub provides, IREE uses
[self-hosted runners](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners)
to run many of its workflow jobs. These enable access to additional compute and
custom configurations such as accelerators. Configuration scripting is checked
in to this repository (see the
[README for that directory](https://github.com/openxla/iree/blob/main/build_tools/github_actions/runner/README.md)).

### Custom managed runners

In addition to our self-hosted runners, we use GitHub's
[large managed runners](https://docs.github.com/en/actions/using-github-hosted-runners/about-larger-runners)
for some platforms that are more trouble to configure ourselves (e.g. Mac).

### CI behavior manipulation

The setup step of the CI determines which CI jobs to run. This is controlled by
the [configure_ci.py](https://github.com/openxla/iree/blob/main/build_tools/github_actions/configure_ci.py) script. It
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
options). See the
[benchmark suites documentation](../performance/benchmark_suites.md).

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

![ci-extra](./contributing_ci-extra.png)

The enabled jobs can be viewed from the Summary page of an action run:

![ci_enabled_jobs](./contributing_ci_enabled_jobs.png)

## Contributor tips

These are opinionated tips documenting workflows that some members of the team
have found useful. They are focused on meta-tooling, not on IREE code
specifically (you will find the latter in the
[Developer Overview](./developer_overview.md)).

!!! note

    It is certainly possible to use workflows other than these. Some common
    tasks, especially for maintainers, will likely be made easier by using
    these flows.

We assume a basic knowledge
of `git` and GitHub and suggests some specific ways of using it.

### Useful tools

* GitHub CLI (<https://github.com/cli/cli>). A CLI for interacting with GitHub.
    Most importantly, it allows scripting the creation of pull requests.
* Refined GitHub Chrome and Firefox Extension:
    <https://github.com/sindresorhus/refined-github>. Nice extension that adds a
    bunch of features to the GitHub UI.
* VSCode: <https://code.visualstudio.com/>. The most commonly used IDE amongst
    IREE developers.
* [Ccache](https://ccache.dev/), a fast C/C++ compiler cache. See our
  [CMake with `ccache`](../building/cmake_with_ccache.md) page

### Git structure

We tend to use the "triangular" or "forking" workflow. Develop primarily on a
clone of the repository on your development machine. Any local branches named
the same as persistent branches from the
[main repository](https://github.com/openxla/iree) are pristine (though
potentially stale) copies. You only fastforward these to match upstream and
otherwise do development on other branches. When sending PRs, you push to a
different branch on your public fork and create the PR from there.

<!-- TODO(scotttodd): screenshots / diagrams here
  (https://mermaid.js.org/syntax/gitgraph.html?) -->

#### Setup

1. Create a fork of the main repository.

2. Create a local git repository with remotes `upstream` (the main repository)
    and `origin` (your personal fork). To list your current remotes
    `git remote -v`.

    a. If you already cloned from the main repository (e.g. by following the
    getting started guide):

    ```shell
    # From your existing git repo
    $ git remote rename origin upstream
    $ git remote add origin https://github.com/<github_username>/iree.git
    ```

    b. If you haven't already cloned:

    ```shell
    # From whatever directory under which you want to nest your repo
    $ git clone https://github.com/<github_username>/iree.git
    $ cd iree
    $ git remote add upstream https://github.com/openxla/iree.git
    ```

    This is especially important for maintainers who have write access (so can
    push directly to the main repository) and admins who have elevated
    privileges (so can push directly to protected branches). These names are
    just suggestions, but you might find some scripts where the defaults are for
    remotes named like this. For extra safety, you can make it difficult to push
    directly to upstream by setting the push url to something invalid: `git
    remote set-url --push upstream DISABLE`, which requires re-enabling the push
    URL explicitly before pushing. You can wrap this behavior in a custom git
    command like
    [git-sudo](https://gist.github.com/GMNGeoffrey/42dd9a9792390094a43bdb69659320c0).

3. Use a script like
    [git_update.sh](https://github.com/openxla/iree/blob/main/build_tools/scripts/git/git_update.sh)
    to easily synchronize `main` with `upstream`. Submodules make this is a
    little trickier than it should be. You can also turn this into a git command
    by adding it to your path as `git-update`.

#### Git config

These are some additional options you could put in your top-level `.gitconfig`
or repository-specific `.git/config` files that are conducive the recommended
workflow

```ini
[push]
  default = current
[alias]
  # Delete branches that you pushed and have been deleted upstream, e.g. because
  # the PR was merged.
  gone = ! "git fetch -p  && git for-each-ref --format '%(refname:short) %(upstream:track)' | awk '$2 == \"[gone]\" {print $1}' | xargs -r git branch -D"
  # Update from upstream (custom command) and delete obsolete local branches.
  sync = ! (git update main && git gone)
  # Create a new branch based off of main (requires a clean working directory).
  new = "!f(){ \\\ngit checkout main && git switch -c $1; \\\n}; f"
  # Display branches in a useful "latest last" format
  br = for-each-ref --sort=committerdate refs/heads/ --format='%(HEAD) %(color:yellow)%(refname:short)%(color:reset) - %(color:red)%(objectname:short)%(color:reset) - %(contents:subject) (%(color:green)%(committerdate:relative)%(color:reset))'
  # `git git foo` -> `git foo` typo fixer
  git = "!f(){ \\\n git \"$@\"; \\\n}; f"
  # Get the git root directory
  root = rev-parse --show-toplevel
  # checkout, but also sync submodules
  ch = "!f() { \\\n git checkout \"$@\"; git submodule sync && git submodule update --init; \\\n}; f"
  # See the diff for a PR branch vs the main branch
  diffmain = diff --merge-base main
  # See only the files that differ vs the main branch
  whatsout = diffmain --name-only
[checkout]
  # If the checkout command
  defaultRemote = origin
[pull]
  # When pulling, only complete the pull if its a clean fast forward.
  ff = only
[remote]
  # Push to your fork (origin) by default
  pushDefault = origin
[url "ssh://git@github.com/"]
  # Pull with https (so no auth required), but push with ssh.
  pushInsteadOf = https://github.com/
```
