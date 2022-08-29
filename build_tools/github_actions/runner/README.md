# GitHub Self-Hosted Runner Configuration

This directory contains configuration for setting up IREE's GitHub Actions
[self-hosted runners](https://docs.github.com/en/actions/hosting-your-own-runners/about-self-hosted-runners).

The [`gcp/`](./gcp) directory contains scripts specific to setting up runners on
Google Cloud Platform (GCP). These are
[Managed Instance Groups](https://cloud.google.com/compute/docs/instance-groups)
that execute the [GitHub actions runner](https://github.com/actions/runner) as a
service initialized on startup. The scripts automate the creation of VM
[Instance Templates](http://cloud/compute/docs/instance-templates) and the
creation and update of the instance groups. These scripts are currently very
early stage and require editing to execute for different configurations rather
than taking flags. They mostly just automate some manual tasks and minimize
errors. Our GCP project is
[iree-oss](https://console.cloud.google.com/?project=iree-oss).

Included in the `gcp` directory is the [startup script](./gcp/startup_script.sh)
that is configured to run when the VM instance starts up. It pulls in the rest
of the configuration from the [`config`](./config) directory at a specified
repository commit.

The [`config/`](./config) directory contains configuration that is pulled into
the runner on VM startup. This configuration registers the runner with the
GitHub Actions control plane and then creates services to start the runner and
to deregister the runner on shutdown. When the runner service exits, it
initiates the shutdown of the VM, which triggers the deregister service.

Also in the config directory is configuration of the runner itself. The entry
point is the [`runner.env`](./config/runner.env) file, which is symlinked into
the runner's `.env` file and directs the runner to run
[hooks before and after each job](https://docs.github.com/en/actions/hosting-your-own-runners/running-scripts-before-or-after-a-job).
We use these hooks to ensure a consistent environment for jobs executed on the
runner and to check that the job was triggered by an event that the runner is
allowed to process (for instance, postsubmit runners will refuse to run a job
triggered by a `pull_request` event).

## Ephemeral Runners and Autoscaling

Our runners are ephemeral, which means that after executing a single job the
runner program exits. As noted above, the runner service triggers a shutdown of
the VM instance when the runner exits. This shutdown triggers the deregister
service which attempts to deregister the runner from the GitHub Actions control
plane. Note that if the runner stopped gracefully (i.e. after completing a job,
it's *supposed* to deregister itself automatically). This deregistration is to
catch other cases. It is best effort (as the instance can execute a non-graceful
shutdown), but the only downside to failed deregistration appears to be
"offline" runner entries hanging around in the UI. GitHub will garbage collect
these after a certain time period (30 days for normal runners and 1 day for
ephemeral runners), so deregistration is not critical.

### Runner Token Proxy

Registering a GitHub Actions Runner requires a registration token. To obtain
such a token, you must have very broad access to either the organization or
repository you are registering it in. This access is too broad to grant to the
runners themselves. Therefore, we mediate the token acquisition through a proxy
hosted on [Google Cloud Run](https://cloud.google.com/run). The proxy has the
app token for a GitHub App with permission to manage self-hosted runners for the
"iree-org" GitHub organization. It receives requests from the runners when they
are trying to register or deregister and returns them the much more narrowly
scoped [de]registration token. We use
https://github.com/google-github-actions/github-runner-token-proxy for the
proxy. You can see its docs for more details.

## Service Accounts

The presubmit and postsubmit runners run as different service accounts depending
on their trust level. Presubmit runners are "minimal" trust and postsubmit
runners are "basic" trust, so they run as
`github-runner-minimal-trust@iree-oss.iam.gserviceaccount.com` and
`github-runner-basic-trust@iree-oss.iam.gserviceaccount.com`, respectively.

## Passing Artifacts

Using GitHub's [artifact actions](https://github.com/actions/upload-artifact)
with runners on GCE turns out to be prohibitively slow (see discussion in
https://github.com/iree-org/iree/issues/9881). Instead we use our own
[Google Cloud Storage](https://cloud.google.com/storage) (GCS) buckets to save
artifacts from jobs and fetch them in subsequent jobs:
`iree-github-actions-presubmit-artifacts` and
`iree-github-actions-postsubmit-artifacts`. Each runner group's service account
has acces only to the bucket for its group. Artifacts are indexed by the
workflow run id and attempt number, so that they do not collide. Subsequent jobs
should *not* make assumptions about where an artifact was stored however,
instead querying the outputs of the job that created it (which should always
provide such an output). This is both to promote DRY principles and for subtle
reasons like a rerun of a failed job may be on run attempt 2, but fetching
artifacts from a job dependency that succeeded on attempt 1 and therefore did
not rerun and recreate the artifacts indexed by the new attempt.

## Labels

The GitHub Actions Runners are identified with
[labels](https://docs.github.com/en/enterprise-cloud@latest/actions/hosting-your-own-runners/using-labels-with-self-hosted-runners)
that indicate properties of the runner. Some of the labels are automatically
generated from information about the runner on startup, such as its GCP zone and
hostname, others match GitHub's standard labels, like the OS, and some are
injected as custom labels via metadata, like whether the VM is optimized for CPU
or GPU usage. All self-hosted runners receive the `self-hosted` label.

Note that when setting where a job runs, any runner that has all the specified
labels can pick up a job. So if you leave off the runner-group, for instance,
the job will non-deterministically try to run on presubmit or postsubmit
runners. We do not currently have a solution for this problem other than careful
code authorship and review.

## Examining Runners

The runners for iree-org can be viewed in the
[GitHub UI](https://github.com/organizations/iree-org/settings/actions/runners).
Unfortunately, only organization admins have access to this page. Organization
admin gives very broad privileges, so this set is necessarily kept very small by
Google security policy.

 ## Updating the Runners

We frequently need to update the runner instances. In particular, after a Runner
release, the version of the program running on the runners must be updated
[within 30 days](https://docs.github.com/en/enterprise-cloud@latest/actions/hosting-your-own-runners/autoscaling-with-self-hosted-runners#controlling-runner-software-updates-on-self-hosted-runners),
otherwise the GitHub control plane will refuse their connection. Testing and
rolling out these updates involves a few steps.

### Test Runners

We have groups of testing runners (tagged with the `environment=testing` label),
that can be used to deploy new runner configurations and can be tested by
targeting jobs using the label. Create templates using the
[`create_templates.sh`](./gcp/create_templates.sh) script, overriding the
`TEMPLATE_CONFIG_REPO` and/or `TEMPLATE_CONFIG_REF` environment variables to
point to your new configurations. Update the testing instance group to your new
template using [`update_instance_group.sh`](./gcp/update_instance_group.sh) (no
need to canary to the test group). The autoscaling configuration for the testing
group usually has both min and max replicas set to 0, so there aren't any
instances running. Update the configuration to something appropriate for your
testing (probably something like 0-10) using
[`update_autoscaling.sh`](./gcp/update_autoscaling.sh). Check that your runners
successfully start up and register with the GitHub UI. Then send a PR or trigger
a workflow dispatch (depending on what you're testing) targeting the testing
environment, and ensure that your new runners work. Send and merge a PR updating
the runner configuration. When you're done, make sure to set the testing group
autoscaling back to 0-0. For now, you'll need to shut down the remaining
runners. The easiest way to do this for now is to go to the UI for the managed
instance group and delete all the instances. The necessity for manual deletion
will go away with future improvements.

### Deploy to Prod

Since the startup script used by the runners references a specific commit,
merging the PR will not immediately affect them. Note that this means that any
changes you make need to be forward and backward compatible with changes to
anything that is picked up directly from tip of tree (such as workflow files).
These should be changed in separate PRs.

To deploy to prod, create new prod templates. Then use the
[`canary_update_instance_group.sh`](./gcp/canary_update_instance_group.sh) script
to canary an update to 10% of runners. Watch to make sure that your new runners
are starting up and registering as expected and there aren't any additional
failures. It is probably best to wait on the order of days before proceeding.

When you are satisfied that your new configuration is good, complete the update
with your new template, using `update_instance_group.sh`.


## Known Issues / Future Work

There are number of known issues and areas of improvement for the runners:

- The only autoscaling currently uses CPU usage (the default), which does not
  work at all for GPU-based runners. The GPU groups are set with minimum and
  maximum autoscaling size set to the same value (this is slightly different
  from being set to a fixed value for detailed reasons that I won't go into). We
  need to set up autoscaling based on
  [GitHub's job queueing webhooks](https://docs.github.com/en/enterprise-cloud@latest/actions/hosting-your-own-runners/autoscaling-with-self-hosted-runners#using-webhooks-for-autoscaling).
- The runners currently use a persistent disk, which results in relatively slow
  IO. Due to errors made in setting these up, the disk image and disk for the
  runners is 1TB, which is also wasteful and results in quota issues.
- If the runner fails to register (e.g. GitHub has a server error, which has
  happened on multiple occasions), the VM will sit idle, running according to
  the autoscaler. The runner doesn't have a any way to check its health status,
  which would allow the autoscaler to recognize there was a problem and replace
  the instance (https://github.com/actions/runner/issues/745).
- MIG autoscaling has the option to scale groups up and down. We currently have
  it set to only scale up. When scaling down, the autoscaler just sends a
  shutdown signal to the instance and it has
  [90 seconds to run a shutdown script](https://cloud.google.com/compute/docs/shutdownscript),
  but can't complete a long-running build. There is no functionality to send a
  gentle shutdown signal. This is especially problematic given that we only have
  CPU-usage based autoscaling at the moment because this is an imperfect measure
  and in particular decides that an instance is idle if it is doing IO (e.g.
  uploading artifacts). Job queue based autoscaling would probably help, but the
  same problem would exist. We likely need to implement functionality in the
  instance to shut itself down after some period of inactivity.
