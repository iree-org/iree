#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import http.client
import requests
import unittest
from unittest import mock
from typing import Any

import post_benchmark_comment


class GithubClientTest(unittest.TestCase):

  def setUp(self):
    self._mock_response = mock.create_autospec(requests.Response)
    self._mock_requester = mock.create_autospec(
        post_benchmark_comment.APIRequester)
    self._mock_requester.get.return_value = self._mock_response
    self._mock_requester.post.return_value = self._mock_response
    self._mock_requester.patch.return_value = self._mock_response

  def test_post_to_gist(self):
    gist_url = "https://example.com/123455/1234.md"
    self._mock_response.status_code = http.client.CREATED
    self._mock_response.json.return_value = {
        "html_url": gist_url,
        "truncated": False
    }
    client = post_benchmark_comment.GithubClient(self._mock_requester)

    url = client.post_to_gist(filename="1234.md", content="xyz")

    self.assertEqual(url, gist_url)
    self._mock_requester.post.assert_called_once_with(
        endpoint=post_benchmark_comment.GITHUB_GIST_API,
        payload={
            "public": True,
            "files": {
                "1234.md": {
                    "content": "xyz"
                }
            }
        })

  def test_post_to_gist_truncated(self):
    gist_url = "example.com/123455/1234.md"
    self._mock_response.status_code = http.client.CREATED
    self._mock_response.json.return_value = {
        "html_url": gist_url,
        "truncated": True
    }
    client = post_benchmark_comment.GithubClient(self._mock_requester)

    with self.assertRaises(RuntimeError) as _:
      client.post_to_gist(filename="1234.md", content="xyz")

  def test_get_previous_comment_on_pr(self):
    first_mock_response = mock.create_autospec(requests.Response)
    first_mock_response.status_code = http.client.OK
    first_mock_response.json.return_value = [{
        "id": 1,
        "user": {
            "login": "bot"
        },
        "body": "comment id: abcd"
    }, {
        "id": 2,
        "user": {
            "login": "user"
        },
        "body": "comment id: 1234"
    }]
    second_mock_response = mock.create_autospec(requests.Response)
    second_mock_response.status_code = http.client.OK
    second_mock_response.json.return_value = [{
        "id": 3,
        "user": {
            "login": "bot"
        },
        "body": "comment id: 1234"
    }]
    mock_requester = mock.create_autospec(post_benchmark_comment.APIRequester)
    mock_requester.get.side_effect = [first_mock_response, second_mock_response]
    client = post_benchmark_comment.GithubClient(mock_requester)

    comment_id = client.get_previous_comment_on_pr(pr_number=23,
                                                   gist_bot_user="bot",
                                                   comment_type_id="1234",
                                                   query_comment_per_page=2,
                                                   max_pages_to_search=10)

    self.assertEqual(comment_id, 3)
    self.assertEqual(mock_requester.get.call_count, 2)
    endpoint_url = f"{post_benchmark_comment.GITHUB_IREE_API_PREFIX}/issues/23/comments"
    mock_requester.get.assert_any_call(endpoint=endpoint_url,
                                       payload={
                                           "per_page": 2,
                                           "page": 1,
                                           "sort": "updated",
                                           "direction": "desc"
                                       })
    mock_requester.get.assert_any_call(endpoint=endpoint_url,
                                       payload={
                                           "per_page": 2,
                                           "page": 2,
                                           "sort": "updated",
                                           "direction": "desc"
                                       })

  def test_get_previous_comment_on_pr_not_found(self):
    mock_response = mock.create_autospec(requests.Response)
    mock_response.status_code = http.client.OK
    mock_response.json.return_value = [{
        "id": 1,
        "user": {
            "login": "bot"
        },
        "body": "comment id: 5678"
    }]
    mock_requester = mock.create_autospec(post_benchmark_comment.APIRequester)
    mock_requester.get.side_effect = [mock_response] * 10
    client = post_benchmark_comment.GithubClient(mock_requester)

    comment_id = client.get_previous_comment_on_pr(pr_number=23,
                                                   gist_bot_user="bot",
                                                   comment_type_id="1234",
                                                   query_comment_per_page=1,
                                                   max_pages_to_search=10)

    self.assertIsNone(comment_id)
    self.assertEqual(mock_requester.get.call_count, 10)
    endpoint_url = f"{post_benchmark_comment.GITHUB_IREE_API_PREFIX}/issues/23/comments"
    mock_requester.get.assert_any_call(endpoint=endpoint_url,
                                       payload={
                                           "per_page": 1,
                                           "page": 1,
                                           "sort": "updated",
                                           "direction": "desc"
                                       })
    mock_requester.get.assert_any_call(endpoint=endpoint_url,
                                       payload={
                                           "per_page": 1,
                                           "page": 10,
                                           "sort": "updated",
                                           "direction": "desc"
                                       })

  def test_update_comment_on_pr(self):
    self._mock_response.status_code = http.client.OK
    client = post_benchmark_comment.GithubClient(self._mock_requester)

    client.update_comment_on_pr(comment_id=123, content="xyz")

    self._mock_requester.patch.assert_called_once_with(
        endpoint=
        f"{post_benchmark_comment.GITHUB_IREE_API_PREFIX}/issues/comments/123",
        payload={"body": "xyz"})

  def test_create_comment_on_pr(self):
    self._mock_response.status_code = http.client.CREATED
    client = post_benchmark_comment.GithubClient(self._mock_requester)

    client.create_comment_on_pr(pr_number=1234, content="xyz")

    self._mock_requester.post.assert_called_once_with(
        endpoint=
        f"{post_benchmark_comment.GITHUB_IREE_API_PREFIX}/issues/1234/comments",
        payload={"body": "xyz"})


if __name__ == "__main__":
  unittest.main()
