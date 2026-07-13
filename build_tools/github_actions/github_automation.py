#!/usr/bin/env python3
#
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
from abc import ABC, abstractmethod
import os

import github

FIRST_TIME_CONTRIBUTOR_COMMENT_TAG = "<!--IREE FIRST TIME CONTRIBUTOR COMMENT-->\n"


class FirstTimeContributorGreeter(ABC):
    @abstractmethod
    def get_first_time_contributor_comment(self, author: str) -> str:
        raise NotImplementedError

    def __init__(self, token: str, repo: str, issue_number: int, author: str):
        self.repo = github.Github(auth=github.Auth.Token(token)).get_repo(repo)
        self.issue = self.repo.get_issue(issue_number)
        self.author = author

    def run(self) -> bool:
        if self.already_commented():
            print(
                f"First-time contributor comment already exists on #{self.issue.number}"
            )
            return True

        self.issue.create_comment(self.get_first_time_contributor_comment(self.author))
        return True

    def already_commented(self) -> bool:
        for comment in self.issue.get_comments():
            if FIRST_TIME_CONTRIBUTOR_COMMENT_TAG in comment.body:
                return True
        return False


class PullRequestGreeter(FirstTimeContributorGreeter):
    def get_first_time_contributor_comment(self, author: str) -> str:
        return f"""\
{FIRST_TIME_CONTRIBUTOR_COMMENT_TAG}
Hello @{author} :wave:

Thank you for submitting a Pull Request to IREE! It looks like this is your first one. Below are some useful links and pointers.

### General guidance
Our general [Contributing guide](https://iree.dev/developers/general/contributing/) contains information and links to detailed guides on code quality, testing, commit summaries and our CI system.

A common point for new PRs: if a DCO signing check fails for you, check out the section on [Developer Certificate of Origin](https://iree.dev/developers/general/contributing/#developer-certificate-of-origin).
In these cases, it should suffice to amend your commit signature(s) per the guide and force-push the PR branch.

### Action required: acknowledge IREE project policies
IREE is a Linux Foundation project. All participants are expected to follow the [LF Projects Code of Conduct](https://lfprojects.org/policies/code-of-conduct/).

Please also note that all contributions to IREE must follow our [IREE AI Tool Use Policy](https://iree.dev/developers/general/contributing/#ai-tool-use). In particular:
- Contributors must fully understand, and vouch for, **all** submitted changes and the intent behind them.
- Contributors should write PR descriptions themselves.
- Substantial use of LLM/generative AI tools must be noted in the PR description, e.g. via `Assisted-by: tool-name` or `Co-authored-by: tool-name tool@email` trailers.
- GitHub issues labeled as "Good first issue" are explicitly designated as learning opportunities for newcomers to the project. We discourage AI tool usage for resolutions to those issues, and substantial use (e.g. fully prompting the fix and/or the tests) is forbidden.

We kindly ask you to **reply to this message** and confirm that you understand and accept the cited policies, particularly the AI Tool Use Policy.

---

If you have any questions, feel free to leave a comment here, or ask away on [IREE Discord](https://discord.gg/e4F4NzX99).

Thank you,
The IREE Community"""


class IssueGreeter(FirstTimeContributorGreeter):
    def get_first_time_contributor_comment(self, author: str) -> str:
        return f"""\
{FIRST_TIME_CONTRIBUTOR_COMMENT_TAG}
Hello @{author} :wave:

Thank you for submitting an issue to IREE! It looks like it's your first one.

The volume of issues is huge, and triage times may vary. If you need quicker assistance on the issue, consider posting it on [IREE Discord](https://discord.gg/e4F4NzX99) and providing short context there.
Feel free to identify the most frequent contributors to the corresponding part of the codebase via Git history/[CODEOWNERS](.github/CODEOWNERS) and (gently!) tag them below.

---

### Action required: acknowledge IREE project policies
IREE is a Linux Foundation project. All participants are expected to follow the [LF Projects Code of Conduct](https://lfprojects.org/policies/code-of-conduct/).

Please note that all contributions to IREE, **including GitHub issues**, must follow our [IREE AI Tool Use Policy](https://iree.dev/developers/general/contributing/#ai-tool-use). This means:
- Issue reporters must fully understand the primary issue and the impact of the problem.
  For bug reports, specifics of the use-case, workflow and/or crash scenario usually suffice.
  For improvement suggestions, this usually implies deeper understanding of suboptimal code, IR design choices, assumed performance tradeoffs, etc.
- Contributors must write the summary of the issue themselves, so that the exact impact and scenario is clear to other participants.
- Issue authors are expected to review and understand all LLM/AI-assisted segments of the report, if any. E.g.: CLI commands and IR reproducers, model/framework details, lowering pipeline details, assumptions about problematic behaviors of IREE components.
- Substantial use of LLM/generative AI tools must be disclaimed in the header or footer of the issue description.
- GitHub issues labeled as "Good first issue" are explicitly designated as learning opportunities for newcomers to the project. We discourage AI tool usage for resolutions to those issues, and substantial use (e.g. fully prompting the fix and/or the tests) is forbidden.

Before the issue can be triaged, we kindly ask you to **reply to this message** and confirm that you understand and accept the proposed policy. As required, please adjust your initial issue description.

---

If you have any questions, feel free to leave a comment here, or ask away on [IREE Discord](https://discord.gg/e4F4NzX99).

Thank you,
The IREE Community"""


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token", type=str, required=True, help="GitHub authentication token"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=os.getenv("GITHUB_REPOSITORY", "iree-org/iree"),
        help="The GitHub repository in the form of <owner>/<repo>",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pr_greeter_parser = subparsers.add_parser("pr-greeter")
    pr_greeter_parser.add_argument("--issue-number", type=int, required=True)
    pr_greeter_parser.add_argument("--author", type=str, required=True)

    issue_greeter_parser = subparsers.add_parser("issue-greeter")
    issue_greeter_parser.add_argument("--issue-number", type=int, required=True)
    issue_greeter_parser.add_argument("--author", type=str, required=True)

    return parser


def main() -> int:
    args = create_argument_parser().parse_args()

    if args.command == "pr-greeter":
        greeter = PullRequestGreeter(
            args.token, args.repo, args.issue_number, args.author
        )
        return 0 if greeter.run() else 1

    if args.command == "issue-greeter":
        greeter = IssueGreeter(args.token, args.repo, args.issue_number, args.author)
        return 0 if greeter.run() else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
