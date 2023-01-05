# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Note that these tests are not run on CI. Doing so would require adding a ton
# of extra dependencies and the frequency of code changes here is not worth the
# extra maintenance burden. Please do run the tests when making changes to the
# service though.

import json
import unittest
from unittest import mock

import google.api_core.exceptions
import werkzeug.exceptions
from google.cloud import compute
from werkzeug.wrappers import Request

import main

# Don't rely on any of the specific values in these
INVALID_TOKEN = "INVALID_TOKEN"
ID1 = 1234
ID2 = 4567
REGION = "us-central1"
ZONE = "us-west1-a"
PROJECT = "iree-oss"
INSTANCE_LINK_PREFIX = "https://www.googleapis.com/compute/v1/projects/iree-oss/zones/us-east1-b/instances/"
INSTANCE_NAME = "some_instance_name"
MIG_PATH_PREFIX = "projects/794014424711/regions/us-north1/instanceGroupManagers/"
MIG_NAME = "some_mig_name"


def get_message(ctx):
  return ctx.exception.get_response().get_data(as_text=True)


# A fake for oauth2 token verification that pretends the encoding scheme is just
# JSON.
def fake_verify_oauth2_token(token, request):
  del request
  return json.loads(token)


def make_token(payload: dict):
  return json.dumps(payload)


@mock.patch("google.oauth2.id_token.verify_oauth2_token",
            fake_verify_oauth2_token)
class InstanceDeleterTest(unittest.TestCase):

  def setUp(self):
    self.addCleanup(mock.patch.stopall)
    instances_client_patcher = mock.patch("main.instances_client",
                                          autospec=True)
    self.instances_client = instances_client_patcher.start()
    migs_client_patcher = mock.patch("main.migs_client", autospec=True)
    self.migs_client = migs_client_patcher.start()
    os_environ_patcher = mock.patch.dict(
        "os.environ", {main.ALLOWED_MIG_PATTERN_ENV_VARIABLE: ".*"})
    self.environ = os_environ_patcher.start()
    autoscalers_client_patcher = mock.patch("main.autoscalers_client",
                                            autospec=True)
    self.autoscalers_client = autoscalers_client_patcher.start()
    time_patcher = mock.patch("time.time", autospec=True)
    self.time = time_patcher.start()
    self.time.return_value = 0
    # Just noop sleep
    mock.patch("time.sleep", autospec=True).start()

  def test_delete_happy_path(self):
    req = Request({}, populate_request=False, shallow=True)
    req.method = "DELETE"

    token = make_token({
        "google": {
            "compute_engine": {
                "project_id": PROJECT,
                "zone": f"{REGION}-a",
                "instance_name": INSTANCE_NAME,
                "instance_id": str(ID1),
            }
        }
    })

    req.headers = {"Authorization": f"Bearer {token}"}

    self_link = f"{INSTANCE_LINK_PREFIX}{INSTANCE_NAME}"
    instance = compute.Instance(
        id=ID1,
        name=INSTANCE_NAME,
        zone=ZONE,
        self_link=self_link,
        metadata=compute.Metadata(items=[
            compute.Items(key=main.MIG_METADATA_KEY,
                          value=f"{MIG_PATH_PREFIX}{MIG_NAME}")
        ]))
    self.instances_client.get.return_value = instance

    response = main.delete_self(req)

    self.assertIn(MIG_NAME, response)
    self.assertIn(INSTANCE_NAME, response)

    self.migs_client.delete_instances.assert_called_once_with(
        instance_group_manager=MIG_NAME,
        project=PROJECT,
        region=REGION,
        region_instance_group_managers_delete_instances_request_resource=compute
        .RegionInstanceGroupManagersDeleteInstancesRequest(
            instances=[instance.self_link]))

  def test_get_happy_path(self):
    req = Request({}, populate_request=False, shallow=True)
    req.method = "GET"

    token = make_token({
        "google": {
            "compute_engine": {
                "project_id": PROJECT,
                "zone": f"{REGION}-a",
                "instance_name": INSTANCE_NAME,
                "instance_id": str(ID1),
            }
        }
    })

    req.headers = {"Authorization": f"Bearer {token}"}

    self_link = f"{INSTANCE_LINK_PREFIX}{INSTANCE_NAME}"
    instance = compute.Instance(
        id=ID1,
        name=INSTANCE_NAME,
        zone=ZONE,
        self_link=self_link,
        metadata=compute.Metadata(items=[
            compute.Items(key=main.MIG_METADATA_KEY,
                          value=f"{MIG_PATH_PREFIX}{MIG_NAME}")
        ]))
    self.instances_client.get.return_value = instance

    mig = compute.InstanceGroupManager(
        target_size=5,
        status={
            "is_stable": True,
            "autoscaler": "autoscaler_link/autoscaler_name"
        })
    self.migs_client.get.return_value = mig

    autoscaler = compute.Autoscaler(recommended_size=3)
    self.autoscalers_client.get.return_value = autoscaler

    response = main.delete_self(req)

    self.assertEqual(response, "true")

  def test_get_timeout(self):
    req = Request({}, populate_request=False, shallow=True)
    req.method = "GET"

    token = make_token({
        "google": {
            "compute_engine": {
                "project_id": PROJECT,
                "zone": f"{REGION}-a",
                "instance_name": INSTANCE_NAME,
                "instance_id": str(ID1),
            }
        }
    })

    req.headers = {"Authorization": f"Bearer {token}"}

    self_link = f"{INSTANCE_LINK_PREFIX}{INSTANCE_NAME}"
    instance = compute.Instance(
        id=ID1,
        name=INSTANCE_NAME,
        zone=ZONE,
        self_link=self_link,
        metadata=compute.Metadata(items=[
            compute.Items(key=main.MIG_METADATA_KEY,
                          value=f"{MIG_PATH_PREFIX}{MIG_NAME}")
        ]))
    self.instances_client.get.return_value = instance

    mig = compute.InstanceGroupManager(
        target_size=5,
        status={
            "is_stable": False,
            "autoscaler": "autoscaler_link/autoscaler_name"
        })
    self.migs_client.get.return_value = mig
    self.time.side_effect = [0, main.STABILIZE_TIMEOUT_SECONDS + 1]

    with self.assertRaises(werkzeug.exceptions.GatewayTimeout):
      main.delete_self(req)

  def test_narrow_allowed_migs(self):
    req = Request({}, populate_request=False, shallow=True)
    req.method = "DELETE"

    token = make_token({
        "google": {
            "compute_engine": {
                "project_id": PROJECT,
                "zone": f"{REGION}-a",
                "instance_name": INSTANCE_NAME,
                "instance_id": str(ID1),
            }
        }
    })

    req.headers = {"Authorization": f"Bearer {token}"}

    mig_name = "github-runner-foo-bar"
    self.environ[main.ALLOWED_MIG_PATTERN_ENV_VARIABLE] = "github-runner-.*"
    self_link = f"{INSTANCE_LINK_PREFIX}{INSTANCE_NAME}"
    instance = compute.Instance(
        id=ID1,
        name=INSTANCE_NAME,
        zone=ZONE,
        self_link=self_link,
        metadata=compute.Metadata(items=[
            compute.Items(key=main.MIG_METADATA_KEY,
                          value=f"{MIG_PATH_PREFIX}{mig_name}")
        ]))
    self.instances_client.get.return_value = instance

    ext_operation = mock.MagicMock(
        google.api_core.extended_operation.ExtendedOperation)
    ext_operation.result.return_value = None

    response = main.delete_self(req)

    self.assertIn(mig_name, response)
    self.assertIn(INSTANCE_NAME, response)

    self.migs_client.delete_instances.assert_called_once_with(
        instance_group_manager=mig_name,
        project=PROJECT,
        region=REGION,
        region_instance_group_managers_delete_instances_request_resource=compute
        .RegionInstanceGroupManagersDeleteInstancesRequest(
            instances=[instance.self_link]))

  def test_bad_method(self):
    req = Request({}, populate_request=False, shallow=True)
    req.method = "POST"

    with self.assertRaises(werkzeug.exceptions.BadRequest) as ctx:
      main.delete_self(req)

    self.assertIn("Invalid method", get_message(ctx))

  def test_bad_path(self):
    req = Request({}, populate_request=False, shallow=True)
    req.method = "DELETE"
    req.path = "/foo/bar"

    with self.assertRaises(werkzeug.exceptions.BadRequest) as ctx:
      main.delete_self(req)

    self.assertIn("Invalid request path", get_message(ctx))

  def test_missing_header(self):
    req = Request({}, populate_request=False, shallow=True)
    req.method = "DELETE"

    with self.assertRaises(werkzeug.exceptions.Unauthorized) as ctx:
      main.delete_self(req)

    self.assertIn("Authorization header is missing", get_message(ctx))

  def test_malformed_header(self):
    req = Request({}, populate_request=False, shallow=True)
    req.method = "DELETE"
    req.headers = {"Authorization": "UnknownScheme token"}

    with self.assertRaises(werkzeug.exceptions.Unauthorized) as ctx:
      main.delete_self(req)

    self.assertIn("Authorization header does not start", get_message(ctx))

  def test_invalid_token(self):
    req = Request({}, populate_request=False, shallow=True)
    req.method = "DELETE"
    req.headers = {"Authorization": f"Bearer {INVALID_TOKEN}"}

    with self.assertRaises(werkzeug.exceptions.Unauthorized) as ctx:
      main.delete_self(req)

    self.assertIn("token", get_message(ctx))

  def test_bad_token_payload(self):
    req = Request({}, populate_request=False, shallow=True)
    req.method = "DELETE"

    token = make_token({"aud": "localhost"})

    req.headers = {"Authorization": f"Bearer {token}"}

    with self.assertRaises(werkzeug.exceptions.Unauthorized) as ctx:
      main.delete_self(req)

    self.assertIn("token", get_message(ctx))

  def test_nonexistent_instance(self):
    req = Request({}, populate_request=False, shallow=True)
    req.method = "DELETE"

    token = make_token({
        "google": {
            "compute_engine": {
                "project_id": PROJECT,
                "zone": ZONE,
                "instance_name": INSTANCE_NAME,
                "instance_id": str(ID1),
            }
        }
    })

    req.headers = {"Authorization": f"Bearer {token}"}

    self.instances_client.get.side_effect = google.api_core.exceptions.NotFound(
        "Instance not found")

    with self.assertRaises(werkzeug.exceptions.NotFound) as ctx:
      main.delete_self(req)

    self.assertIn(INSTANCE_NAME, get_message(ctx))

  def test_id_mismatch(self):
    req = Request({}, populate_request=False, shallow=True)
    req.method = "DELETE"

    token = make_token({
        "google": {
            "compute_engine": {
                "project_id": PROJECT,
                "zone": ZONE,
                "instance_name": INSTANCE_NAME,
                "instance_id": str(ID1),
            }
        }
    })

    req.headers = {"Authorization": f"Bearer {token}"}

    instance = compute.Instance(id=ID2, name=INSTANCE_NAME)

    self.instances_client.get.return_value = instance

    with self.assertRaises(werkzeug.exceptions.BadRequest) as ctx:
      main.delete_self(req)

    msg = get_message(ctx)
    self.assertIn(str(ID1), msg)
    self.assertIn(str(ID2), msg)

  def test_missing_mig_metadata(self):
    req = Request({}, populate_request=False, shallow=True)
    req.method = "DELETE"

    token = make_token({
        "google": {
            "compute_engine": {
                "project_id": PROJECT,
                "zone": ZONE,
                "instance_name": INSTANCE_NAME,
                "instance_id": str(ID1),
            }
        }
    })

    req.headers = {"Authorization": f"Bearer {token}"}

    instance = compute.Instance(id=ID1,
                                name=INSTANCE_NAME,
                                zone=ZONE,
                                self_link=f"http://foo/bar/{INSTANCE_NAME}")

    self.instances_client.get.return_value = instance

    with self.assertRaises(werkzeug.exceptions.BadRequest) as ctx:
      main.delete_self(req)

    self.assertIn(main.MIG_METADATA_KEY, get_message(ctx))

  def test_mig_pattern_unset(self):
    req = Request({}, populate_request=False, shallow=True)
    req.method = "DELETE"

    token = make_token({
        "google": {
            "compute_engine": {
                "project_id": PROJECT,
                "zone": f"{REGION}-a",
                "instance_name": INSTANCE_NAME,
                "instance_id": str(ID1),
            }
        }
    })

    req.headers = {"Authorization": f"Bearer {token}"}

    self_link = f"{INSTANCE_LINK_PREFIX}{INSTANCE_NAME}"
    instance = compute.Instance(
        id=ID1,
        name=INSTANCE_NAME,
        zone=ZONE,
        self_link=self_link,
        metadata=compute.Metadata(items=[
            compute.Items(key=main.MIG_METADATA_KEY,
                          value=f"{MIG_PATH_PREFIX}{MIG_NAME}")
        ]))
    self.instances_client.get.return_value = instance

    del self.environ[main.ALLOWED_MIG_PATTERN_ENV_VARIABLE]

    with self.assertRaises(werkzeug.exceptions.InternalServerError) as ctx:
      main.delete_self(req)

    self.assertIn(main.ALLOWED_MIG_PATTERN_ENV_VARIABLE, get_message(ctx))

  def test_no_migs_allowed(self):
    req = Request({}, populate_request=False, shallow=True)
    req.method = "DELETE"

    token = make_token({
        "google": {
            "compute_engine": {
                "project_id": PROJECT,
                "zone": f"{REGION}-a",
                "instance_name": INSTANCE_NAME,
                "instance_id": str(ID1),
            }
        }
    })

    req.headers = {"Authorization": f"Bearer {token}"}

    instance = compute.Instance(
        id=ID1,
        name=INSTANCE_NAME,
        zone=ZONE,
        self_link=f"{INSTANCE_LINK_PREFIX}{INSTANCE_NAME}",
        metadata=compute.Metadata(items=[
            compute.Items(key=main.MIG_METADATA_KEY,
                          value=f"{MIG_PATH_PREFIX}{MIG_NAME}")
        ]))
    self.instances_client.get.return_value = instance

    self.environ[main.ALLOWED_MIG_PATTERN_ENV_VARIABLE] = ""

    with self.assertRaises(werkzeug.exceptions.Forbidden) as ctx:
      main.delete_self(req)

    self.assertIn(MIG_NAME, get_message((ctx)))

  def test_mig_not_allowed(self):
    req = Request({}, populate_request=False, shallow=True)
    req.method = "DELETE"

    token = make_token({
        "google": {
            "compute_engine": {
                "project_id": PROJECT,
                "zone": f"{REGION}-a",
                "instance_name": INSTANCE_NAME,
                "instance_id": str(ID1),
            }
        }
    })

    req.headers = {"Authorization": f"Bearer {token}"}

    mig_name = "not-github-runner"
    self.environ[main.ALLOWED_MIG_PATTERN_ENV_VARIABLE] = "github-runner-.*"
    instance = compute.Instance(
        id=ID1,
        name=INSTANCE_NAME,
        zone=ZONE,
        self_link=f"{INSTANCE_LINK_PREFIX}{INSTANCE_NAME}",
        metadata=compute.Metadata(items=[
            compute.Items(key=main.MIG_METADATA_KEY,
                          value=f"{MIG_PATH_PREFIX}{mig_name}")
        ]))
    self.instances_client.get.return_value = instance

    with self.assertRaises(werkzeug.exceptions.Forbidden) as ctx:
      main.delete_self(req)

    self.assertIn(mig_name, get_message((ctx)))

  def test_bad_deletion_request_server(self):
    req = Request({}, populate_request=False, shallow=True)
    req.method = "DELETE"

    token = make_token({
        "google": {
            "compute_engine": {
                "project_id": PROJECT,
                "zone": ZONE,
                "instance_name": INSTANCE_NAME,
                "instance_id": str(ID1),
            }
        }
    })

    req.headers = {"Authorization": f"Bearer {token}"}

    instance = compute.Instance(
        id=ID1,
        name=INSTANCE_NAME,
        zone=ZONE,
        self_link=f"{INSTANCE_LINK_PREFIX}{INSTANCE_NAME}",
        metadata=compute.Metadata(items=[
            compute.Items(key=main.MIG_METADATA_KEY,
                          value=f"{MIG_PATH_PREFIX}{MIG_NAME}")
        ]))

    self.instances_client.get.return_value = instance
    self.migs_client.delete_instances.side_effect = ValueError("Bad request")

    with self.assertRaises(werkzeug.exceptions.InternalServerError) as ctx:
      main.delete_self(req)

    self.assertIn(MIG_NAME, get_message(ctx))

  # Testing of server errors is unimplemented. ExtendedOperation is not
  # documented well enough for me to produce a reasonable fake and a bad fake is
  # worse than nothing.


if __name__ == "__main__":
  unittest.main()
