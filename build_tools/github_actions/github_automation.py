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

Thank you for submitting a pull request to IREE! It looks like this is your first one. We have one ask, and you can also find some general tips below.

---

### Action required: acknowledge the IREE policies and contribution guidelines
IREE is a Linux Foundation project. All participants are expected to follow the [LF Projects Code of Conduct](https://lfprojects.org/policies/code-of-conduct/).

All contributions to IREE must follow our [contribution guidelines](https://github.com/iree-org/iree/blob/main/CONTRIBUTING.md). In particular:
- Contributors own their changes: they must review **all** of them themselves and understand them well enough to explain and defend them. AI review tools do not replace this review.
- Contributors also own their changes after merge: they fix or revert them when problems come up.
- PR descriptions may be AI-assisted, but contributors must review them and follow the [pull request description guidelines](https://github.com/iree-org/iree/blob/main/CONTRIBUTING.md#pull-request-descriptions).
- Contributions are written for human readers: the people who review and maintain them, and who read the history later. AI clutter, such as boilerplate, comments that narrate obvious code, restated diffs, agent logs, or filler text, does not belong in any contribution.
- There must always be a human in the loop: contributors must respond to reviews and questions by themselves. A lack of understanding makes a contribution [extractive](https://github.com/iree-org/iree/blob/main/CONTRIBUTING.md#extractive-contributions), because it puts the burden on the reviewers instead of the author.
- Every contribution, whether a pull request, an issue, or a comment, should be worth more to the project than the time maintainers spend on it. Otherwise it is [extractive](https://github.com/iree-org/iree/blob/main/CONTRIBUTING.md#extractive-contributions) and may be closed. It is then up to the author to show that it is not.
- GitHub issues labeled as "Good first issue" are explicitly designated as learning opportunities for newcomers to the project. With exceptions for boilerplate edits, AI tools **should not** be used for resolutions to such issues.

If this pull request does not follow these guidelines yet, for example because an AI agent opened it without your review, please close it and reopen it once you have reworked the change.

We kindly ask you to **reply to this message** and confirm that you understand and accept the policies and contribution guidelines.

---

### General guidance
The [website version of the contribution guidelines](https://iree.dev/developers/general/contributing/) also covers CI configuration, commit signing, and commit access.

A common point for new PRs: if a DCO signing check fails for you, check out the section on [Developer Certificate of Origin](https://github.com/iree-org/iree/blob/main/CONTRIBUTING.md#developer-certificate-of-origin).
In these cases, it should suffice to amend your commit signature(s) per the guide and force-push the PR branch.

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

---

### Action required: acknowledge the IREE policies and contribution guidelines
IREE is a Linux Foundation project. All participants are expected to follow the [LF Projects Code of Conduct](https://lfprojects.org/policies/code-of-conduct/).

Please note that all contributions to IREE, **including GitHub issues**, must follow our [contribution guidelines](https://github.com/iree-org/iree/blob/main/CONTRIBUTING.md). In particular:
- Issue authors own their report: they must understand the problem and its impact, and review every AI-assisted part, such as reproducers, commands, or analysis.
- The description may be AI-assisted, but it must make the scenario and impact clear to others. For a bug, this is the use case, workflow, or crash. For an improvement, it is the affected code and the expected benefit.
- Issues are written for human readers: the maintainers and contributors who triage and fix them. AI clutter, such as boilerplate, agent logs, or filler text, does not belong in the report.
- There must always be a human in the loop: issue authors answer follow-up questions themselves.
- Every contribution, whether a pull request, an issue, or a comment, should be worth more to the project than the time maintainers spend on it. Otherwise it is [extractive](https://github.com/iree-org/iree/blob/main/CONTRIBUTING.md#extractive-contributions) and may be closed. It is then up to the author to show that it is not.

Before the issue can be triaged, we kindly ask you to **reply to this message** and confirm that you understand and accept the policies and contribution guidelines. If needed, update the issue description accordingly.

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
