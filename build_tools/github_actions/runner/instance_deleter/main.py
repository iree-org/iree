# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""A Cloud Functions proxy enabling GCE VMs in a Managed Instance Group to delete themselves.

GCE Managed instance groups don't have any good way to handle autoscaling for
long-running workloads. With the autoscaler configured to scale in, instances
get only 90 seconds warning to shut down. So we set the autoscaler to only scale
out and have the VMs tear themselves down when they're down with their work.
This is the approach suggested by the managed instance group team:

https://drive.google.com/file/d/1XlwxF_0T7pUnbzhL5ePDoW-Q3GAaLO11

But anything that brings down the VM other than a delete call to the instance
group manager API makes the VM get considered "unhealthy", which means it gets
recreated in exactly the same configuration, regardless of any update or
autoscaling settings. Making the correct API call requires broad permissions on
the instance group manager, which we don't want to give the VMs. To scope
permissions to individual instances, this proxy service makes use of instance
identity tokens to allow an instance to make a call only to delete itself.

See
https://cloud.google.com/compute/docs/instances/verifying-instance-identity

This makes use of the GCP Cloud Functions serverless offering. It's another
level of abstraction on top of Cloud Run, where you don't even need to create your
own docker container. For local development:

  functions-framework --target=delete_self
  curl -X DELETE -v --header "Authorization: Bearer $(cat /tmp/token.txt)" localhost:8080

You'll need to get a token that corresponds to an actual instance though or
you'll get an error:

  gcloud compute ssh gh-runner-testing-presubmit-cpu-us-west1-h58j \
    --user-output-enabled=false \
    --command "curl -sSfL \
        -H 'Metadata-Flavor: Google' \
        'http://metadata/computeMetadata/v1/instance/service-accounts/default/identity?audience=localhost&format=full'" \
    > /tmp/token.txt

To deploy:
  # Note timeout should be greater than STABILIZE_TIMEOUT_SECONDS
  gcloud functions deploy instance-self-deleter \
    --gen2 \
    --runtime=python310 \
    --region=us-central1 \
    --source=. \
    --entry-point=delete_self \
    --trigger-http \
    --run-service-account=managed-instance-deleter@iree-oss.iam.gserviceaccount.com \
    --service-account=managed-instance-deleter@iree-oss.iam.gserviceaccount.com \
    --ingress-settings=internal-only \
    --timeout=120s \
    --set-env-vars ALLOWED_MIG_PATTERN='gh-runner-.*'


See https://cloud.google.com/functions/docs for more details.
"""

import os
import random
import re
import time
from http.client import (BAD_REQUEST, FORBIDDEN, GATEWAY_TIMEOUT,
                         INTERNAL_SERVER_ERROR, NOT_FOUND, UNAUTHORIZED)

import flask
import functions_framework
import google.api_core.exceptions
import google.auth.exceptions
import requests
from google.auth import transport
from google.cloud import compute
from google.oauth2 import id_token

AUTH_HEADER_PREFIX = "Bearer "
ALLOWED_HTTP_METHODS = ["DELETE", "GET"]
MIG_METADATA_KEY = "created-by"
ALLOWED_MIG_PATTERN_ENV_VARIABLE = "ALLOWED_MIG_PATTERN"
# Must be less than timeout configuration for deployment
STABILIZE_TIMEOUT_SECONDS = 100

instances_client = compute.InstancesClient()
migs_client = compute.RegionInstanceGroupManagersClient()
autoscalers_client = compute.RegionAutoscalersClient()
session = requests.Session()

print("Server started")


def _verify_token(token: str) -> dict:
  """Verify token signature and return the token payload"""
  request = transport.requests.Request(session)
  payload = id_token.verify_oauth2_token(token, request=request)
  return payload


def _get_region(zone: str) -> str:
  """Extract region name from zone name"""
  # Drop the trailing zone identifier to get the region. Yeah it kinda does seem
  # like there should be a better way to do this...
  region, _ = zone.rsplit("-", maxsplit=1)
  return region


def _get_name_from_resource(resource: str) -> str:
  """Extract just the final name component from a fully scoped resource name."""
  _, name = resource.rsplit("/", maxsplit=1)
  return name


def _get_from_items(items: compute.Items, key: str):
  # Why would the GCP Python API return something as silly as a dictionary?
  return next((item.value for item in items if item.key == key), None)


def delete_instance_from_mig(mig_name: str, project: str, region: str,
                             instance: compute.Instance):
  try:
    operation = migs_client.delete_instances(
        instance_group_manager=mig_name,
        project=project,
        region=region,
        # For some reason we can't just use a list of instance names and need to
        # build this RhymingRythmicJavaClasses proto. Also, unlike all the other
        # parameters, the instance has to be a fully-specified URL for the
        # instance, not just its name.
        region_instance_group_managers_delete_instances_request_resource=(
            compute.RegionInstanceGroupManagersDeleteInstancesRequest(
                instances=[instance.self_link])))
  except (google.api_core.exceptions.Forbidden,
          google.api_core.exceptions.Unauthorized,
          google.api_core.exceptions.NotFound) as e:
    print(e)
    return flask.abort(
        e.code, f"Error requesting that {mig_name} delete {instance.name}.")
  except Exception as e:
    # We'll call any other error here a server error.
    print(e)
    return flask.abort(
        INTERNAL_SERVER_ERROR,
        f"Error requesting that {mig_name} delete {instance.name}.")

  try:
    # This is actually an extended operation that you have to poll to get its
    # status, but we just check the status once because it appears that errors
    # always show up here and all we just want to return success in marking for
    # deletion. We don't need to wait for the deletion to actually take place.
    operation.result()
  except google.api_core.exceptions.ClientError as e:
    print(e)
    # Unpack the actual usable error message
    msg = (
        f"Error requesting that {mig_name} delete {instance.name}:"
        "\n" + "\n".join(
            [f"{err.code}: {err.message}" for err in e.response.error.errors]))
    print(msg)
    # We're not actually totally sure whether this is a client or server error
    # for the overall request, but let's call it a client error (the only client
    # here is our VM instances, so I think we can be a bit loose).
    return flask.abort(BAD_REQUEST, msg)

  success_msg = f"{instance.name} has been marked for deletion by {mig_name}."
  print(success_msg)
  return success_msg


def should_scale_down(mig_name: str, project: str, region: str):
  start = time.time()
  print(f"Polling {mig_name} for stability")
  while time.time() - start < STABILIZE_TIMEOUT_SECONDS:
    try:
      mig = migs_client.get(project=project,
                            region=region,
                            instance_group_manager=mig_name)
    except google.api_core.exceptions.NotFound as e:
      print(e)
      return flask.abort(
          e.code,
          f"Cannot find {mig_name} in region={region}, project={project}")
    if mig.status.is_stable:
      break
    # We sleep for a random amount of time here to avoid synchronizing callers
    # waiting for the MIG to be stable.
    sleep_secs = random.randint(1, 15)
    print(f"{mig_name} is not stable. Retrying in {sleep_secs} seconds")
    time.sleep(sleep_secs)
  else:
    return flask.abort(GATEWAY_TIMEOUT,
                       "Timed out waiting for the MIG to become stable")
  autoscaler = autoscalers_client.get(project=project,
                                      region=region,
                                      autoscaler=_get_name_from_resource(
                                          mig.status.autoscaler))
  response = "true" if autoscaler.recommended_size < mig.target_size else "false"
  print(
      f"Autoscaler recommends size {autoscaler.recommended_size} and"
      f" {mig_name} is targetting size {mig.target_size}. Sending: {response}")
  return response


@functions_framework.http
def delete_self(request: flask.Request):
  """HTTP Cloud Function to delete the instance group making the request.
    Args:
        request: The request object.
        https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response.
    Note:
        For more information on how Flask integrates with Cloud
        Functions, see the `Writing HTTP functions` page.
        https://cloud.google.com/functions/docs/writing/http#http_frameworks
  """
  if request.method not in ALLOWED_HTTP_METHODS:
    return flask.abort(
        BAD_REQUEST, f"Invalid method {request.method}."
        f" Allowed methods: {ALLOWED_HTTP_METHODS}")

  # No path is needed, since the token and method contain all the information we
  # need. Maybe that design was a mistake, but since the resource being operated
  # on is always the instance making the call, it seemed handy.
  if request.path != "/":
    return flask.abort(
        BAD_REQUEST,
        f"Invalid request path {request.path}. Only root path is valid).")

  auth_header = request.headers.get("Authorization")
  if auth_header is None:
    return flask.abort(UNAUTHORIZED, "Authorization header is missing")
  if not auth_header.startswith(AUTH_HEADER_PREFIX):
    return flask.abort(
        UNAUTHORIZED,
        f"Authorization header does not start with expected string"
        f" {AUTH_HEADER_PREFIX}.")

  token = auth_header[len(AUTH_HEADER_PREFIX):]

  try:
    # We don't verify audience here because Cloud IAM will have already done so
    # and jwt's matching of audiences is exact, which means trailing slashes or
    # http vs https matters and that's pretty brittle.
    token_payload = _verify_token(token)
  except (ValueError, google.auth.exceptions.GoogleAuthError) as e:
    print(e)
    return flask.abort(UNAUTHORIZED, "Decoding bearer token failed.")

  print(f"Token payload: {token_payload}")

  try:
    compute_info = token_payload["google"]["compute_engine"]
  except KeyError:
    return flask.abort(
        UNAUTHORIZED,
        "Bearer token payload does not have expected field google.compute")

  project = compute_info["project_id"]
  zone = compute_info["zone"]
  region = _get_region(zone)
  instance_name = compute_info["instance_name"]

  if request.method == "DELETE":
    print(f"Received request to delete {instance_name}")
  else:
    assert request.method == "GET"
    print(f"Received inquiry whether to delete {instance_name}")
  try:
    instance = instances_client.get(instance=instance_name,
                                    project=project,
                                    zone=zone)
  except (google.api_core.exceptions.NotFound,
          google.api_core.exceptions.Forbidden) as e:
    print(e)
    return flask.abort(
        e.code,
        f"Cannot view {instance_name} in zone={zone}, project={project}")

  instance_id = int(compute_info["instance_id"])
  # Verify it's *actually* the same instance. Names get reused, but IDs don't.
  # For some reason you can't reference anything by ID in the API.
  if instance.id != instance_id:
    return flask.abort(
        BAD_REQUEST,
        f"Existing instance of the same name {instance.name} has a different"
        f" ID {instance.id} than token specifies {instance_id}.")

  mig_name = _get_from_items(instance.metadata.items, MIG_METADATA_KEY)

  if mig_name is None:
    return flask.abort(BAD_REQUEST,
                       (f"Instance is not part of a managed instance group."
                        f" Did not find {MIG_METADATA_KEY} in metadata."))
  mig_name = _get_name_from_resource(mig_name)

  # General good practice would be to compile the regex once, but the only way
  # to do that is to make it a global, which makes this difficult to test and
  # compiling this regex should not be expensive.
  allowed_mig_pattern = os.environ.get(ALLOWED_MIG_PATTERN_ENV_VARIABLE)
  if allowed_mig_pattern is None:
    flask.abort(
        INTERNAL_SERVER_ERROR, f"Missing required environment variable"
        f" {ALLOWED_MIG_PATTERN_ENV_VARIABLE}")

  if not re.fullmatch(allowed_mig_pattern, mig_name):
    return flask.abort(FORBIDDEN, f"No access to MIG {mig_name}")

  if request.method == "DELETE":
    return delete_instance_from_mig(mig_name=mig_name,
                                    project=project,
                                    region=region,
                                    instance=instance)

  assert request.method == "GET"
  return should_scale_down(mig_name=mig_name, project=project, region=region)
