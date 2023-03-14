#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Posts benchmark results to gist and comments on pull requests.

Requires the environment variables:

- GITHUB_TOKEN: token from GitHub action that has write access on issues. See
  https://docs.github.com/en/actions/security-guides/automatic-token-authentication#permissions-for-the-github_token
- COMMENT_BOT_USER: user name that posts the comment. Note this can be different
    from the user creates the gist.
- GIST_BOT_TOKEN: token that has write access to gist. Gist will be posted as
  the owner of the token. See
  https://docs.github.com/en/rest/overview/permissions-required-for-fine-grained-personal-access-tokens#gists
"""

import sys
import pathlib

# Add build_tools python dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.with_name("python")))

import argparse
import http.client
import json
import os
import requests
from typing import Any, Optional

from reporting import benchmark_comment

GITHUB_IREE_API_PREFIX = "https://api.github.com/repos/openxla/iree"
GITHUB_GIST_API = "https://api.github.com/gists"
GITHUB_API_VERSION = "2022-11-28"


class APIRequester(object):
  """REST API client that injects proper GitHub authentication headers."""

  def __init__(self, github_token: str):
    self._api_headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"token {github_token}",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    self._session = requests.session()

  def get(self, endpoint: str, payload: Any = {}) -> requests.Response:
    return self._session.get(endpoint,
                             data=json.dumps(payload),
                             headers=self._api_headers)

  def post(self, endpoint: str, payload: Any = {}) -> requests.Response:
    return self._session.post(endpoint,
                              data=json.dumps(payload),
                              headers=self._api_headers)

  def patch(self, endpoint: str, payload: Any = {}) -> requests.Response:
    return self._session.patch(endpoint,
                               data=json.dumps(payload),
                               headers=self._api_headers)


class GithubClient(object):
  """Helper to call Github REST APIs."""

  def __init__(self, requester: APIRequester):
    self._requester = requester

  def post_to_gist(self,
                   filename: str,
                   content: str,
                   verbose: bool = False) -> str:
    """Posts the given content to a new GitHub Gist and returns the URL to it."""

    response = self._requester.post(endpoint=GITHUB_GIST_API,
                                    payload={
                                        "public": True,
                                        "files": {
                                            filename: {
                                                "content": content
                                            }
                                        }
                                    })
    if response.status_code != http.client.CREATED:
      raise RuntimeError(
          f"Failed to create on gist; error code: {response.status_code} - {response.text}"
      )

    response = response.json()
    if verbose:
      print(f"Gist posting response: {response}")

    if response["truncated"]:
      raise RuntimeError(f"Content is too large and was truncated")

    return response["html_url"]

  def get_previous_comment_on_pr(self,
                                 pr_number: int,
                                 comment_bot_user: str,
                                 comment_type_id: str,
                                 query_comment_per_page: int = 100,
                                 max_pages_to_search: int = 10,
                                 verbose: bool = False) -> Optional[int]:
    """Gets the previous comment's id from GitHub."""

    for page in range(1, max_pages_to_search + 1):
      response = self._requester.get(
          endpoint=f"{GITHUB_IREE_API_PREFIX}/issues/{pr_number}/comments",
          payload={
              "per_page": query_comment_per_page,
              "page": page,
              "sort": "updated",
              "direction": "desc"
          })
      if response.status_code != http.client.OK:
        raise RuntimeError(
            f"Failed to get PR comments from GitHub; error code: {response.status_code} - {response.text}"
        )

      comments = response.json()
      if verbose:
        print(f"Previous comment query response on page {page}: {comments}")

      # Find the most recently updated comment that matches.
      for comment in comments:
        if (comment["user"]["login"] == comment_bot_user and
            comment_type_id in comment["body"]):
          return comment["id"]

      if len(comments) < query_comment_per_page:
        break

    return None

  def update_comment_on_pr(self, comment_id: int, content: str):
    """Updates the content of the given comment id."""

    response = self._requester.patch(
        endpoint=f"{GITHUB_IREE_API_PREFIX}/issues/comments/{comment_id}",
        payload={"body": content})
    if response.status_code != http.client.OK:
      raise RuntimeError(
          f"Failed to comment on GitHub; error code: {response.status_code} - {response.text}"
      )

  def create_comment_on_pr(self, pr_number: int, content: str):
    """Posts the given content as comments to the current pull request."""

    response = self._requester.post(
        endpoint=f"{GITHUB_IREE_API_PREFIX}/issues/{pr_number}/comments",
        payload={"body": content})
    if response.status_code != http.client.CREATED:
      raise RuntimeError(
          f"Failed to comment on GitHub; error code: {response.status_code} - {response.text}"
      )

  def get_pull_request_head_commit(self, pr_number: int) -> str:
    """Get pull request head commit SHA."""

    response = self._requester.get(
        endpoint=f"{GITHUB_IREE_API_PREFIX}/pulls/{pr_number}")
    if response.status_code != http.client.OK:
      raise RuntimeError(
          f"Failed to fetch the pull request: {pr_number}; "
          f"error code: {response.status_code} - {response.text}")

    return response.json()["head"]["sha"]


def _parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("comment_json", type=pathlib.Path)
  parser.add_argument("--verbose", action="store_true")
  verification_parser = parser.add_mutually_exclusive_group(required=True)
  verification_parser.add_argument("--github_event_json", type=pathlib.Path)
  # Temporary option for buildkite pipeline.
  verification_parser.add_argument("--no_verify_pr", action="store_true")
  return parser.parse_args()


def main(args: argparse.Namespace):
  github_token = os.environ.get("GITHUB_TOKEN")
  if github_token is None:
    raise ValueError("GITHUB_TOKEN must be set.")

  comment_bot_user = os.environ.get("COMMENT_BOT_USER")
  if comment_bot_user is None:
    raise ValueError("COMMENT_BOT_USER must be set.")

  gist_bot_token = os.environ.get("GIST_BOT_TOKEN")
  if gist_bot_token is None:
    raise ValueError("GIST_BOT_TOKEN must be set.")

  comment_data = benchmark_comment.CommentData(
      **json.loads(args.comment_json.read_text()))
  # Sanitize the pr number to make sure it is an integer.
  pr_number = int(comment_data.unverified_pr_number)

  pr_client = GithubClient(requester=APIRequester(github_token=github_token))
  if args.github_event_json is None:
    github_event = None
  else:
    github_event = json.loads(args.github_event_json.read_text())
    workflow_run_sha = github_event["workflow_run"]["head_sha"]
    pr_head_sha = pr_client.get_pull_request_head_commit(pr_number=pr_number)
    # We can't get the trusted PR number of a workflow run from GitHub API. So we
    # take the untrusted PR number from presubmit workflow and verify if the PR's
    # current head SHA matches the commit SHA in the workflow run. It assumes
    # that to generate the malicious comment data, attacker must modify the code
    # and has a new commit SHA. So if the PR head commit matches the workflow
    # run with attacker's commit, either the PR is created by the attacker or
    # other's PR has the malicious commit. In both cases posting malicious
    # comment is acceptable.
    #
    # Note that the collision of a target SHA1 is possible but GitHub has some
    # protections (https://github.blog/2017-03-20-sha-1-collision-detection-on-github-com/).
    # The assumption also only holds if files in GCS can't be overwritten (so the
    # comment data can't be modified without changing the code).
    # The check will also fail if the PR author pushes the new commit after the
    # workflow is triggered. But pushing the new commit means to cancel the
    # current CI run including the benchmarking. So it will unlikely fail for
    # that reason.
    if workflow_run_sha != pr_head_sha:
      raise ValueError(
          f"Workflow run SHA: {workflow_run_sha} does not match "
          f"the head SHA: {pr_head_sha} of the pull request: {pr_number}.")

  gist_client = GithubClient(requester=APIRequester(
      github_token=gist_bot_token))
  gist_url = gist_client.post_to_gist(
      filename=f'iree-full-benchmark-results-{pr_number}.md',
      content=comment_data.full_md,
      verbose=args.verbose)

  previous_comment_id = pr_client.get_previous_comment_on_pr(
      pr_number=pr_number,
      comment_bot_user=comment_bot_user,
      comment_type_id=comment_data.type_id,
      verbose=args.verbose)

  abbr_md = comment_data.abbr_md.replace(
      benchmark_comment.GIST_LINK_PLACEHORDER, gist_url)
  if github_event is not None:
    abbr_md += f'\n\n[Source Workflow Run]({github_event["workflow_run"]["html_url"]})'
  if previous_comment_id is not None:
    pr_client.update_comment_on_pr(comment_id=previous_comment_id,
                                   content=abbr_md)
  else:
    pr_client.create_comment_on_pr(pr_number=pr_number, content=abbr_md)


if __name__ == "__main__":
  main(_parse_arguments())
