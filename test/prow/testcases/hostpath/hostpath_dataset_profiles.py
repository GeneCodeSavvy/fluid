"""
TestCase: HostPath dataset profile tests - stress different data patterns
DDC Engine: Alluxio
Profiles:
  - small_files: 100K+ small files (1KB each) to stress metadata and fuse open/close
  - medium_bin: Sequential read of medium chunks (TFRecord-style, ~1GB total)
  - large_bin: Few very large files (multi-GB) to stress cache capacity and throughput

Steps per profile:
1. create synthetic data on host via setup pod
2. create Dataset(local://) & AlluxioRuntime with nodeAffinity
3. check if dataset is bound
4. run cold read job (no prior cache)
5. run warm read job (cache populated from cold read)
6. collect timing and throughput metrics
7. clean up
"""

import os
import sys
import time


project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

import fluid.fluidapi as fluidapi
import fluid.step_funcs as funcs
from framework.testflow import TestFlow
from framework.step import (
    SimpleStep,
    StatusCheckStep,
    dummy_back,
    currying_fn,
)
from framework.exception import TestError

from kubernetes import client, config


# Dataset profile definitions
PROFILES = {
    "small_files": {
        "description": "Lots of small files (stress metadata and fuse open/close)",
        "setup_script": (
            "set -e; "
            "mkdir -p /hostdata/small_files; "
            "echo 'Generating 10000 x 1KB files...'; "
            "for i in $(seq 1 10000); do "
            "  dd if=/dev/urandom of=/hostdata/small_files/file_${i}.dat bs=1024 count=1 2>/dev/null; "
            "done; "
            "echo 'Done: 10000 files'; "
            "ls /hostdata/small_files | wc -l"
        ),
        "read_script": (
            "set -ex; "
            "echo '=== Cold read: small files ==='; "
            "start=$(date +%s); "
            "find /data -type f | xargs -n 100 cat > /dev/null 2>&1; "
            "end=$(date +%s); "
            'echo "METRIC cold_read_seconds=$((end - start))"; '
            "file_count=$(find /data -type f | wc -l); "
            'echo "METRIC file_count=${file_count}"'
        ),
        "warm_read_script": (
            "set -ex; "
            "echo '=== Warm read: small files ==='; "
            "start=$(date +%s); "
            "find /data -type f | xargs -n 100 cat > /dev/null 2>&1; "
            "end=$(date +%s); "
            'echo "METRIC warm_read_seconds=$((end - start))"'
        ),
        "hostpath_dir": "/mnt/data/hostpath-profile-small",
        "cache_quota": "2Gi",
    },
    "medium_bin": {
        "description": "Medium-size binary files (TFRecord-style sequential reads)",
        "setup_script": (
            "set -e; "
            "mkdir -p /hostdata/medium_bin; "
            "echo 'Generating 10 x 100MB files...'; "
            "for i in $(seq 1 10); do "
            "  dd if=/dev/urandom of=/hostdata/medium_bin/chunk_${i}.bin bs=1048576 count=100 2>/dev/null; "
            "done; "
            "echo 'Done'; "
            "du -sh /hostdata/medium_bin"
        ),
        "read_script": (
            "set -ex; "
            "echo '=== Cold read: medium bin ==='; "
            "start=$(date +%s); "
            'for f in /data/medium_bin/*.bin; do cat "$f" > /dev/null; done; '
            "end=$(date +%s); "
            'echo "METRIC cold_read_seconds=$((end - start))"; '
            "total=$(du -sm /data/medium_bin | cut -f1); "
            "elapsed=$((end - start)); "
            "if [ $elapsed -gt 0 ]; then "
            "  throughput=$((total / elapsed)); "
            '  echo "METRIC throughput_mb_per_sec=${throughput}"; '
            "fi"
        ),
        "warm_read_script": (
            "set -ex; "
            "echo '=== Warm read: medium bin ==='; "
            "start=$(date +%s); "
            'for f in /data/medium_bin/*.bin; do cat "$f" > /dev/null; done; '
            "end=$(date +%s); "
            'echo "METRIC warm_read_seconds=$((end - start))"; '
            "total=$(du -sm /data/medium_bin | cut -f1); "
            "elapsed=$((end - start)); "
            "if [ $elapsed -gt 0 ]; then "
            "  throughput=$((total / elapsed)); "
            '  echo "METRIC warm_throughput_mb_per_sec=${throughput}"; '
            "fi"
        ),
        "warm_read_script": (
            "set -ex; "
            "echo '=== Warm read: medium bin ==='; "
            "start=$(date +%s); "
            'for f in /data/*.bin; do cat "$f" > /dev/null; done; '
            "end=$(date +%s); "
            'echo "METRIC warm_read_seconds=$((end - start))"; '
            "total=$(du -sm /data | cut -f1); "
            "elapsed=$((end - start)); "
            "if [ $elapsed -gt 0 ]; then "
            "  throughput=$((total / elapsed)); "
            '  echo "METRIC warm_throughput_mb_per_sec=${throughput}"; '
            "fi"
        ),
        "hostpath_dir": "/mnt/data/hostpath-profile-medium",
        "cache_quota": "4Gi",
    },
    "large_bin": {
        "description": "Large binary files (stress cache capacity, eviction, throughput)",
        "setup_script": (
            "set -e; "
            "mkdir -p /hostdata/large_bin; "
            "echo 'Generating 2 x 1GB files...'; "
            "dd if=/dev/urandom of=/hostdata/large_bin/large_1.bin bs=1048576 count=1024 2>/dev/null; "
            "dd if=/dev/urandom of=/hostdata/large_bin/large_2.bin bs=1048576 count=1024 2>/dev/null; "
            "echo 'Done'; "
            "du -sh /hostdata/large_bin"
        ),
        "read_script": (
            "set -ex; "
            "echo '=== Cold read: large bin ==='; "
            "start=$(date +%s); "
            'for f in /data/large_bin/*.bin; do cat "$f" > /dev/null; done; '
            "end=$(date +%s); "
            'echo "METRIC cold_read_seconds=$((end - start))"; '
            "total=$(du -sm /data/large_bin | cut -f1); "
            "elapsed=$((end - start)); "
            "if [ $elapsed -gt 0 ]; then "
            "  throughput=$((total / elapsed)); "
            '  echo "METRIC throughput_mb_per_sec=${throughput}"; '
            "fi"
        ),
        "warm_read_script": (
            "set -ex; "
            "echo '=== Warm read: large bin ==='; "
            "start=$(date +%s); "
            'for f in /data/large_bin/*.bin; do cat "$f" > /dev/null; done; '
            "end=$(date +%s); "
            'echo "METRIC warm_read_seconds=$((end - start))"; '
            "total=$(du -sm /data/large_bin | cut -f1); "
            "elapsed=$((end - start)); "
            "if [ $elapsed -gt 0 ]; then "
            "  throughput=$((total / elapsed)); "
            '  echo "METRIC warm_throughput_mb_per_sec=${throughput}"; '
            "fi"
        ),
        "warm_read_script": (
            "set -ex; "
            "echo '=== Warm read: large bin ==='; "
            "start=$(date +%s); "
            'for f in /data/*.bin; do cat "$f" > /dev/null; done; '
            "end=$(date +%s); "
            'echo "METRIC warm_read_seconds=$((end - start))"; '
            "total=$(du -sm /data | cut -f1); "
            "elapsed=$((end - start)); "
            "if [ $elapsed -gt 0 ]; then "
            "  throughput=$((total / elapsed)); "
            '  echo "METRIC warm_throughput_mb_per_sec=${throughput}"; '
            "fi"
        ),
        "hostpath_dir": "/mnt/data/hostpath-profile-large",
        "cache_quota": "8Gi",
    },
}


def create_setup_pod(profile_name, profile, target_node):
    """Create a pod to generate synthetic data on the host."""
    api = client.CoreV1Api()

    hostdata_parent = os.path.dirname(profile["hostpath_dir"])
    hostdata_subdir = os.path.basename(profile["hostpath_dir"])

    pod = client.V1Pod(
        api_version="v1",
        kind="Pod",
        metadata=client.V1ObjectMeta(name="hostpath-setup-{}".format(profile_name)),
        spec=client.V1PodSpec(
            node_name=target_node,
            restart_policy="Never",
            containers=[
                client.V1Container(
                    name="setup",
                    image="busybox",
                    command=["/bin/sh"],
                    args=["-c", profile["setup_script"]],
                    volume_mounts=[
                        client.V1VolumeMount(
                            name="host-vol",
                            mount_path="/hostdata",
                            sub_path=hostdata_subdir,
                        )
                    ],
                )
            ],
            volumes=[
                client.V1Volume(
                    name="host-vol",
                    host_path=client.V1HostPathVolumeSource(
                        path=hostdata_parent, type="DirectoryOrCreate"
                    ),
                )
            ],
        ),
    )

    api.create_namespaced_pod(namespace="default", body=pod)
    print(
        "Setup pod for profile '{}' created on node '{}'".format(
            profile_name, target_node
        )
    )


def check_setup_pod_completed_fn(profile_name):
    def check():
        api = client.CoreV1Api()
        pod_name = "hostpath-setup-{}".format(profile_name)
        try:
            pod = api.read_namespaced_pod(name=pod_name, namespace="default")
            if pod.status.phase == "Succeeded":
                return True
            if pod.status.phase == "Failed":
                raise TestError("Setup pod {} failed".format(pod_name))
        except client.exceptions.ApiException as e:
            if e.status == 404:
                raise TestError("Setup pod {} not found".format(pod_name))
        return False

    return check


def delete_setup_pod_fn(profile_name):
    def delete():
        api = client.CoreV1Api()
        pod_name = "hostpath-setup-{}".format(profile_name)
        try:
            api.delete_namespaced_pod(
                name=pod_name,
                namespace="default",
                body=client.V1DeleteOptions(propagation_policy="Background"),
            )
        except client.exceptions.ApiException as e:
            if e.status != 404:
                raise

    return delete


def get_target_node(label_key, label_value):
    """Get the first node matching the given label or label the first available node."""
    api = client.CoreV1Api()

    # Check if any node already has the label
    nodes = api.list_node(label_selector="{}={}".format(label_key, label_value))
    if nodes.items:
        return nodes.items[0].metadata.name

    # Label the first node
    all_nodes = api.list_node()
    if not all_nodes.items:
        raise TestError("No nodes available in cluster")

    node_name = all_nodes.items[0].metadata.name
    api.patch_node(
        name=node_name, body={"metadata": {"labels": {label_key: label_value}}}
    )
    print("Labeled node '{}' with {}={}".format(node_name, label_key, label_value))
    return node_name


def run_profile_test(profile_name, profile):
    """Run a complete test flow for one dataset profile."""
    namespace = "default"
    name = "hostpath-{}".format(profile_name.replace("_", "-"))
    node_label_key = "fluid.io/hostpath-test"
    node_label_value = "true"
    job_name_cold = "hostpath-cold-{}".format(profile_name.replace("_", "-"))
    job_name_warm = "hostpath-warm-{}".format(profile_name.replace("_", "-"))

    target_node = get_target_node(node_label_key, node_label_value)

    # Build Dataset and Runtime
    mount = fluidapi.Mount()
    mount.set_mount_info(name, "local://{}".format(profile["hostpath_dir"]))

    dataset = fluidapi.Dataset(name, namespace)
    dataset.add_mount(mount.dump())
    dataset.set_node_affinity(node_label_key, node_label_value)

    runtime = fluidapi.Runtime("AlluxioRuntime", name, namespace)
    runtime.set_replicas(1)
    runtime.set_tieredstore("MEM", "/dev/shm", profile["cache_quota"])

    flow = TestFlow(
        "HostPath Profile - {} ({})".format(profile_name, profile["description"])
    )

    # Step 1: Setup synthetic data on host
    flow.append_step(
        SimpleStep(
            step_name="create setup pod for '{}'".format(profile_name),
            forth_fn=currying_fn(
                create_setup_pod,
                profile_name=profile_name,
                profile=profile,
                target_node=target_node,
            ),
            back_fn=delete_setup_pod_fn(profile_name),
        )
    )

    flow.append_step(
        StatusCheckStep(
            step_name="wait for data setup to complete",
            forth_fn=check_setup_pod_completed_fn(profile_name),
            timeout=600,
            interval=5,
        )
    )

    # Step 2: Create Dataset and Runtime
    flow.append_step(
        SimpleStep(
            step_name="create dataset",
            forth_fn=funcs.create_dataset_fn(dataset.dump()),
            back_fn=funcs.delete_dataset_and_runtime_fn(
                runtime.dump(), name, namespace
            ),
        )
    )

    flow.append_step(
        SimpleStep(
            step_name="create runtime",
            forth_fn=funcs.create_runtime_fn(runtime.dump()),
            back_fn=dummy_back,
        )
    )

    flow.append_step(
        StatusCheckStep(
            step_name="check if dataset is bound",
            forth_fn=funcs.check_dataset_bound_fn(name, namespace),
        )
    )

    flow.append_step(
        StatusCheckStep(
            step_name="check if PV & PVC is ready",
            forth_fn=funcs.check_volume_resource_ready_fn(name, namespace),
        )
    )

    # Step 3: Cold read
    flow.append_step(
        SimpleStep(
            step_name="create cold read job",
            forth_fn=funcs.create_job_fn(
                profile["read_script"], name, name=job_name_cold, namespace=namespace
            ),
            back_fn=funcs.delete_job_fn(name=job_name_cold, namespace=namespace),
        )
    )

    flow.append_step(
        StatusCheckStep(
            step_name="check cold read job completed",
            forth_fn=funcs.check_job_status_fn(name=job_name_cold, namespace=namespace),
            timeout=600,
            interval=5,
        )
    )

    # Step 4: Warm read (cache should be populated from cold read)
    flow.append_step(
        SimpleStep(
            step_name="create warm read job",
            forth_fn=funcs.create_job_fn(
                profile["warm_read_script"],
                name,
                name=job_name_warm,
                namespace=namespace,
            ),
            back_fn=funcs.delete_job_fn(name=job_name_warm, namespace=namespace),
        )
    )

    flow.append_step(
        StatusCheckStep(
            step_name="check warm read job completed",
            forth_fn=funcs.check_job_status_fn(name=job_name_warm, namespace=namespace),
            timeout=600,
            interval=5,
        )
    )

    return flow


def main():
    if os.getenv("KUBERNETES_SERVICE_HOST") is None:
        config.load_kube_config()
    else:
        config.load_incluster_config()

    # Run the profile specified by env var, or all profiles
    target_profile = os.getenv("HOSTPATH_PROFILE", "all")

    if target_profile == "all":
        profiles_to_run = PROFILES
    elif target_profile in PROFILES:
        profiles_to_run = {target_profile: PROFILES[target_profile]}
    else:
        print(
            "Unknown profile '{}'. Available: {}".format(
                target_profile, ", ".join(PROFILES.keys())
            )
        )
        exit(1)

    failed = False
    for profile_name, profile in profiles_to_run.items():
        print("\n" + "=" * 60)
        print("Running profile: {} - {}".format(profile_name, profile["description"]))
        print("=" * 60 + "\n")

        flow = run_profile_test(profile_name, profile)
        try:
            flow.run()
        except Exception as e:
            print("Profile '{}' FAILED: {}".format(profile_name, e))
            failed = True

    if failed:
        exit(1)


if __name__ == "__main__":
    main()
