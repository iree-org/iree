# Contributing to IREE

<!-- --8<-- [start:intro] -->
We'd love to accept your patches and contributions to this project.

<!-- --8<-- [end:intro] -->

This file covers the policies that apply to every contributor. The
[Contributing guide](https://iree.dev/developers/general/contributing/) on the
website contains the same text and adds details for maintainers and frequent
contributors, such as commit access and CI configuration.

<!-- --8<-- [start:getting-in-touch] -->
## Getting in touch

* [GitHub issues](https://github.com/iree-org/iree/issues): Feature requests,
  bugs, and other work tracking
* [IREE Discord server](https://discord.gg/wEWh6Z9nMU): Daily development
  discussions with maintainers, contributors, and collaborators
* [iree-technical-discussion email list](https://lists.lfaidata.foundation/g/iree-technical-discussion):
  General and low-priority discussion

Please [file issues](https://github.com/iree-org/iree/issues/new/choose) or
reach out on any of these channels before doing substantial work; this will
ensure that others don't duplicate the work and that there's a chance to
discuss any design issues.

<!-- --8<-- [end:getting-in-touch] -->

<!-- --8<-- [start:developer-policies] -->
## Developer policies

### Code of conduct

This project follows the
[LF Projects code of conduct](https://lfprojects.org/policies/code-of-conduct/).

### Project governance

IREE is a sandbox-stage project of the
[LF AI & Data Foundation](https://lfaidata.foundation/). Technical
oversight of the project is the responsibility of its Technical
Steering Committee (TSC), as set out in the
[IREE Technical Charter](https://github.com/lfai/foundation/blob/main/technical%20project%20charters/IREE%20Technical%20Charter%20Final%205-14-2024.docx.pdf).

As provided for in section 2.b of the charter, the voting members of
the TSC are:

* Maximilian Bartel (@maxbartel), RooflineAI — Chairperson
* Tobias Fuchs (@devtbi), RooflineAI
* Artem Gindinson (@AGindinson), RooflineAI
* Stella Laurenzo (@stellaraccident), AMD
* Christopher McGirr (@chrsmcgrr), RooflineAI
* Stefan Schuermans (@schuermans-roofline), RooflineAI
* Jelle Schuhmacher (@jschuhmacher), RooflineAI
* Ben Vanik (@benvanik), AMD

See [MAINTAINERS.md](https://github.com/iree-org/iree/blob/main/MAINTAINERS.md)
for how the TSC operates. TSC meetings are open to everyone.

### Developer Certificate of Origin

Contributors must certify that they wrote or otherwise have the right to submit
the code they are contributing to the project, by agreeing to the
[Developer Certificate of Origin](https://developercertificate.org/) (DCO).

<details markdown="1">
<summary>Expand to read the full DCO agreement text</summary>

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

</details>

Signing is enforced by the [DCO GitHub App](https://github.com/apps/dco) (see
also the [dcoapp/app](https://github.com/dcoapp/app) repository).

The DCO check requires that all commits included in pull requests _either_
are cryptographically signed by a member of the repository's organization _or_
include a `Signed-off-by` message as a git trailer. Frequent contributors can
[sign their commits cryptographically](https://iree.dev/developers/general/contributing/#cryptographically-signing-commits).

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

<!-- --8<-- [end:developer-policies] -->

<!-- --8<-- [start:coding-policies] -->
## Coding policies

### Coding style guidelines

Our formatting rules are enforced by language-specific formatters like
`clang-format` for C++ and `black` for Python. In addition to formatting, we
follow these coding standards:

#### Compiler

The C++ compiler portion of the project follows the
[MLIR style guide](https://mlir.llvm.org/getting_started/DeveloperGuide/#style-guide)
based on the [LLVM coding standards](https://llvm.org/docs/CodingStandards.html).
We also follow the recommendations of the
[LLVM Programmer's Manual](https://llvm.org/docs/ProgrammersManual.html).

IREE deviates from the MLIR style guide in the following ways:

* We use braces with single-line `if` statements and loops.
* We allow for `static` functions in anonymous namespaces to avoid repeatedly
  reopening/closing namespaces.
* Pass _declarations_ in TableGen (`.td`) files should use a `Pass` suffix
  (e.g., `def FooBarPass`), while the corresponding _implementation_ file
  should not include the `Pass` suffix (e.g., `FooBar.cpp`, not
  `FooBarPass.cpp`).

#### Runtime

Most of the code style is derived from the
[Google Style Guides](http://google.github.io/styleguide/) for the appropriate
language.

#### Other

For code outside of the `compiler/` and `runtime/` subdirectories, follow the
style used in existing files.

#### Formatting and Linting

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

Improvements to code structure and clarity are welcome but please file issues
to track such work first. Pure style changes are unlikely to be accepted
unless they are applied consistently across the project.

### Testing policy

With few exceptions, features should be accompanied by automated tests.

We use a mix of in-tree and out-of-tree unit and integration tests. For more
information about the types of tests used across the project, refer to the
[testing guide](https://iree.dev/developers/general/testing-guide/).

<!-- --8<-- [end:coding-policies] -->

<!-- --8<-- [start:github-policies] -->
## GitHub policies

### Code reviews

All submissions, including submissions by maintainers, require review. We
use GitHub pull requests (PRs) for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

* Please keep PRs small (focused on a single issue or feature) to make reviews
  and later culprit-finding easier.
    - You may see trusted core contributors bending above rule for project
      maintenance and major subsystem renovation. If you feel like the rules
      aren't working for a certain situation, please ask as we bias towards
      pragmatism for cases that require it.
* We require review comments to be addressed before merging, even when there is
  an approval from another reviewer. Unless stated otherwise, we treat request
  for changes as blocking. Reviewers may indicate that some comments
  requesting changes are optional (commonly by prefixing them with 'nit' or
  'optional') -- use your best engineering judgement to decide whether to
  follow these suggestions or not.
* If there are new comments requesting changes after a PR was merged, these
  comments should be addressed as well, either through follow-up discussion
  or code changes. Small issues may be fixed forward, but reviewers may also
  ask for your PR to be reverted. This is not a sign that your PR was faulty
  -- we prefer to err on the side of caution and reverting allows more time
  for design discussion.

### GitHub Actions workflows

We use [GitHub Actions](https://docs.github.com/en/actions) to automatically
build and test various parts of the project.

* Most presubmit workflows will only run automatically on PRs if you are a
  project collaborator. Otherwise a maintainer must
  [approve workflow runs](https://docs.github.com/en/actions/managing-workflow-runs/approving-workflow-runs-from-public-forks).
  If you are sending code changes to the project, please
  [request commit access](https://iree.dev/developers/general/contributing/#obtaining-commit-access),
  so that these can run automatically.
* It is generally expected that PRs will only be merged when all checks are
  passing. In some cases, pre-existing failures may be bypassed by a maintainer.

Some workflows only run on commits after they are merged. See
[CI behavior manipulation](https://iree.dev/developers/general/contributing/#ci-behavior-manipulation)
to learn how to customize this behavior.

<!-- --8<-- [end:github-policies] -->

<!-- --8<-- [start:github-policies-continued] -->
### Pull request descriptions

PRs are [squash-merged](#merging-approved-changes), so the PR title and
description become the commit message, and the commit messages on the PR
branch are discarded. Write them for someone reading `git log` or bisecting
a regression later, and update them when the PR changes during review.

**Title**: a short, imperative summary of what changes and, if it fits, why,
prefixed with the affected area, for example
`[CI] Bump iree-test-suites to fix sharktank test failure`. Keep it under 72
characters. Avoid ambiguous titles like "Fix bug" or "Fix CI".

**Description**: start with _why_, then _what_.

1. **Why**: The most important part of a PR description.
   Explain the underlying motivation of the PR. Why did you create it and what is the use-case this enables or fixes.
   Be concise. Depending on the change, this covers:
    * for a bug fix, the symptom, its impact, and the issue or failing run
    * for a regression fix, also the commit or PR that introduced it
    * for a feature, the missing capability and who needs it
    * for a refactor, the current limitation
    * for a performance change, the bottleneck, with numbers
    * for a revert or reland, what broke or what was fixed since, and the
      original PR
2. **Change**: what the PR does and why this approach. The diff shows the
   details. For refactors, state "No functional change intended". For small
   PRs, one sentence is enough.
3. **If relevant**:
    * results: before/after numbers for performance, memory, or size
      changes, and how they were measured
    * trade-offs: known shortcomings, costs, or side effects
    * testing: only tests that CI does not cover
    * review notes: where reviewers should start in large PRs
4. **Trailers** at the end: `Fixes #1234` for issues that the PR resolves,
   `Depends on #1234` for PRs that it builds on,
   [CI trailers](https://iree.dev/developers/general/contributing/#ci-behavior-manipulation)
   such as `ci-extra:`, and `Assisted-by:` for [AI tools](#ai-tool-use).

Keep the description concise and self-contained: links add context but do
not replace an explanation, and they must be public. Leave out restatements
of the diff, implementation details that belong in code comments, review
history, and `@`-mentions, which notify people again on every cherry-pick.
A description that keeps growing is a sign that the PR should be split.
Small changes need one or two sentences; longer descriptions can use
headings such as `## Why`, `## Change`, and `## Results`.

See also:

* [LLVM: Commit messages](https://llvm.org/docs/DeveloperPolicy.html#commit-messages)
* [Google Engineering Practices: Writing good CL descriptions](https://google.github.io/eng-practices/review/developer/cl-descriptions.html)
* [Linux kernel: Describe your changes](https://docs.kernel.org/process/submitting-patches.html#describe-your-changes)
* [GitHub: Helping others review your changes](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/helping-others-review-your-changes)
* [How to write a Git commit message](https://cbea.ms/git-commit/)

### Merging approved changes

After review and presubmit checks, PRs should typically be merged using
"squash and merge".

* The squashed commit summary should match the PR title and the commit
  description should match the PR body (this is the default behavior).
  Accordingly, please write these as you would a helpful commit message
  (see [Pull request descriptions](#pull-request-descriptions)).

It is assumed that the PR author will merge their change unless they ask
someone else to merge it for them (e.g. because they don't have write access
yet).

### AI tool use

We adopt the [LLVM AI Tool Use Policy](https://llvm.org/docs/AIToolPolicy.html)
and require all AI-assisted contributions (Pull Requests, issues, design
proposals) to be reviewed and understood by the author(s).

The key points include:

1. "Contributors must read and review all LLM-generated code or text before they
   ask other project members to review it."
2. "Contributors are expected to be transparent and label contributions that
   contain substantial amounts of tool-generated content." This can be done with
   the `Assisted-by: tool-name` or the `Co-authored-by: tool-name tool@email`
   trailer in PR descriptions.
3. "To ensure sufficient self review and understanding of the work, it is
   strongly recommended that contributors write PR descriptions themselves (if
   needed, using tools for translation or copy-editing)." This also extends to
   code documentation, issue descriptions, and design proposals.
4. "Contributors are responsible for ensuring that they have the right to
   contribute code under the terms of our license, [...]. Using AI tools to
   regenerate copyrighted material does not remove the copyright, and
   contributors are responsible for ensuring that such material does not appear
   in their contributions."

We extend the LLVM policy with an exception for AI bot-authored contributions
(e.g., updating dependencies, fixing build script, etc.); bot account may create
Pull Requests as long as the bot maintainers ensure they are first in line and
primarily responsible for reviewing those contributions.

<!-- --8<-- [end:github-policies-continued] -->
