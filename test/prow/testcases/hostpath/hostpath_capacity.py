"""
TestCase: HostPath single-node capacity test
DDC Engine: Alluxio

Goal: Determine how much dataset load a single node can handle before saturation.
Strategy: Progressively increase concurrent datasets or workloads on the same node,
collecting node metrics at each step.

Metrics collected:
  - Node CPU and memory usage (via kubectl top)
  - Runtime (master/worker/fuse) pod resource usage
  - Job completion time per iteration
  - OOMKills or pod restarts

Environment variables:
  HOSTPATH_CAPACITY_MAX_DATASETS: Max number of concurrent datasets to deploy (default: 5)
  HOSTPATH_CAPACITY_DATA_SIZE_MB: Size of synthetic data per dataset in MB (default: 100)
  HOSTPATH_CAPACITY_NODE_LABEL_KEY: Node label key (default: fluid.io/hostpath-test)
  HOSTPATH_CAPACITY_NODE_LABEL_VALUE: Node label value (default: true)
"""
import os
import sys
import time
import json

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

import fluid.fluidapi as fluidapi
import fluid.step_funcs as funcs
from framework.testflow import TestFlow
from framework.step import SimpleStep, StatusCheckStep, dummy_back, currying_fn
from framework.exception import TestError

from kubernetes import client, config


def get_env_int(key, default):
    val = os.getenv(key)
    return int(val) if val else default


MAX_DATASETS = get_env_int("HOSTPATH_CAPACITY_MAX_DATASETS", 5)
DATA_SIZE_MB = get_env_int("HOSTPATH_CAPACITY_DATA_SIZE_MB", 100)
NODE_LABEL_KEY = os.getenv("HOSTPATH_CAPACITY_NODE_LABEL_KEY", "fluid.io/hostpath-test")
NODE_LABEL_VALUE = os.getenv("HOSTPATH_CAPACITY_NODE_LABEL_VALUE", "true")
HOSTPATH_BASE = "/mnt/data/hostpath-capacity"


def collect_node_metrics(node_name):
    """Collect node-level metrics via the Kubernetes metrics API."""
    metrics = {"node": node_name, "timestamp": time.time()}

    try:
        api = client.CustomObjectsApi()
        node_metrics = api.get_cluster_custom_object(
            group="metrics.k8s.io",
            version="v1beta1",
            plural="nodes",
            name=node_name
        )
        metrics["cpu_usage"] = node_metrics.get("usage", {}).get("cpu", "unknown")
        metrics["memory_usage"] = node_metrics.get("usage", {}).get("memory", "unknown")
    except Exception as e:
        print("Warning: could not collect node metrics: {}".format(e))
        metrics["cpu_usage"] = "unavailable"
        metrics["memory_usage"] = "unavailable"

    return metrics


def collect_pod_metrics(namespace, label_selector):
    """Collect pod-level metrics for runtime components."""
    pod_metrics_list = []
    try:
        api = client.CustomObjectsApi()
        all_pod_metrics = api.list_namespaced_custom_object(
            group="metrics.k8s.io",
            version="v1beta1",
            plural="pods",
            namespace=namespace
        )

        core_api = client.CoreV1Api()
        pods = core_api.list_namespaced_pod(namespace=namespace, label_selector=label_selector)
        pod_names = {p.metadata.name for p in pods.items}

        for item in all_pod_metrics.get("items", []):
            if item["metadata"]["name"] in pod_names:
                containers = []
                for c in item.get("containers", []):
                    containers.append({
                        "name": c["name"],
                        "cpu": c.get("usage", {}).get("cpu", "unknown"),
                        "memory": c.get("usage", {}).get("memory", "unknown"),
                    })
                pod_metrics_list.append({
                    "pod": item["metadata"]["name"],
                    "containers": containers,
                })
    except Exception as e:
        print("Warning: could not collect pod metrics: {}".format(e))

    return pod_metrics_list


def check_for_oom_kills(namespace):
    """Check if any pods in the namespace have been OOMKilled."""
    api = client.CoreV1Api()
    oom_pods = []

    pods = api.list_namespaced_pod(namespace=namespace)
    for pod in pods.items:
        if pod.status and pod.status.container_statuses:
            for cs in pod.status.container_statuses:
                if cs.last_state and cs.last_state.terminated:
                    if cs.last_state.terminated.reason == "OOMKilled":
                        oom_pods.append({
                            "pod": pod.metadata.name,
                            "container": cs.name,
                            "restart_count": cs.restart_count,
                        })

    return oom_pods


def get_target_node():
    """Get or label a target node."""
    api = client.CoreV1Api()

    nodes = api.list_node(label_selector="{}={}".format(NODE_LABEL_KEY, NODE_LABEL_VALUE))
    if nodes.items:
        return nodes.items[0].metadata.name

    all_nodes = api.list_node()
    if not all_nodes.items:
        raise TestError("No nodes available in cluster")

    node_name = all_nodes.items[0].metadata.name
    api.patch_node(
        name=node_name,
        body={"metadata": {"labels": {NODE_LABEL_KEY: NODE_LABEL_VALUE}}}
    )
    print("Labeled node '{}' with {}={}".format(node_name, NODE_LABEL_KEY, NODE_LABEL_VALUE))
    return node_name


def create_setup_pod_for_dataset(idx, target_node):
    """Create synthetic data on host for dataset idx."""
    api = client.CoreV1Api()
    hostpath_dir = "{}/ds{}".format(HOSTPATH_BASE, idx)
    pod_name = "cap-setup-{}".format(idx)

    setup_script = (
        "set -e; "
        "mkdir -p /hostdata; "
        "dd if=/dev/urandom of=/hostdata/data.bin bs=1048576 count={} 2>/dev/null; "
        "echo 'Setup done for dataset {}'; "
        "du -sh /hostdata"
    ).format(DATA_SIZE_MB, idx)

    pod = client.V1Pod(
        api_version="v1",
        kind="Pod",
        metadata=client.V1ObjectMeta(name=pod_name),
        spec=client.V1PodSpec(
            node_name=target_node,
            restart_policy="Never",
            containers=[
                client.V1Container(
                    name="setup",
                    image="busybox",
                    command=["/bin/sh"],
                    args=["-c", setup_script],
                    volume_mounts=[
                        client.V1VolumeMount(name="host-vol", mount_path="/hostdata")
                    ]
                )
            ],
            volumes=[
                client.V1Volume(
                    name="host-vol",
                    host_path=client.V1HostPathVolumeSource(
                        path=hostpath_dir,
                        type="DirectoryOrCreate"
                    )
                )
            ]
        )
    )

    api.create_namespaced_pod(namespace="default", body=pod)
    print("Setup pod '{}' created for dataset {} on node '{}'".format(pod_name, idx, target_node))


def check_setup_pod_fn(idx):
    def check():
        api = client.CoreV1Api()
        pod_name = "cap-setup-{}".format(idx)
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


def delete_setup_pod_fn(idx):
    def delete():
        api = client.CoreV1Api()
        try:
            api.delete_namespaced_pod(
                name="cap-setup-{}".format(idx),
                namespace="default",
                body=client.V1DeleteOptions(propagation_policy="Background")
            )
        except client.exceptions.ApiException as e:
            if e.status != 404:
                raise
    return delete


def run_capacity_iteration(idx, target_node):
    """Deploy one dataset + runtime + read job and return metrics."""
    namespace = "default"
    ds_name = "cap-ds-{}".format(idx)
    hostpath_dir = "{}/ds{}".format(HOSTPATH_BASE, idx)
    job_name = "cap-job-{}".format(idx)

    mount = fluidapi.Mount()
    mount.set_mount_info(ds_name, "local://{}".format(hostpath_dir))

    dataset = fluidapi.Dataset(ds_name, namespace)
    dataset.add_mount(mount.dump())
    dataset.set_node_affinity(NODE_LABEL_KEY, NODE_LABEL_VALUE)

    runtime = fluidapi.Runtime("AlluxioRuntime", ds_name, namespace)
    runtime.set_replicas(1)
    quota_mb = min(DATA_SIZE_MB * 2, 4096)
    runtime.set_tieredstore("MEM", "/dev/shm", "{}Mi".format(quota_mb))

    read_script = (
        "set -ex; "
        "echo '=== Capacity read for dataset {} ==='; "
        "start=$(date +%s); "
        "cat /data/data.bin > /dev/null; "
        "end=$(date +%s); "
        "echo \"METRIC read_seconds=$((end - start))\""
    ).format(idx)

    flow = TestFlow("Capacity iteration {} - {}MB dataset".format(idx, DATA_SIZE_MB))

    # Setup data
    flow.append_step(
        SimpleStep(
            step_name="setup data for dataset {}".format(idx),
            forth_fn=currying_fn(
                create_setup_pod_for_dataset,
                idx=idx,
                target_node=target_node
            ),
            back_fn=delete_setup_pod_fn(idx)
        )
    )
    flow.append_step(
        StatusCheckStep(
            step_name="wait setup pod {}".format(idx),
            forth_fn=check_setup_pod_fn(idx),
            timeout=300,
            interval=5
        )
    )

    # Create Dataset + Runtime
    flow.append_step(
        SimpleStep(
            step_name="create dataset {}".format(idx),
            forth_fn=funcs.create_dataset_fn(dataset.dump()),
            back_fn=funcs.delete_dataset_and_runtime_fn(runtime.dump(), ds_name, namespace)
        )
    )
    flow.append_step(
        SimpleStep(
            step_name="create runtime {}".format(idx),
            forth_fn=funcs.create_runtime_fn(runtime.dump()),
            back_fn=dummy_back
        )
    )
    flow.append_step(
        StatusCheckStep(
            step_name="check dataset {} bound".format(idx),
            forth_fn=funcs.check_dataset_bound_fn(ds_name, namespace),
            timeout=300
        )
    )

    # Read job
    flow.append_step(
        SimpleStep(
            step_name="create read job {}".format(idx),
            forth_fn=funcs.create_job_fn(read_script, ds_name, name=job_name, namespace=namespace),
            back_fn=funcs.delete_job_fn(name=job_name, namespace=namespace)
        )
    )
    flow.append_step(
        StatusCheckStep(
            step_name="check read job {} completed".format(idx),
            forth_fn=funcs.check_job_status_fn(name=job_name, namespace=namespace),
            timeout=600,
            interval=5
        )
    )

    return flow


def main():
    if os.getenv("KUBERNETES_SERVICE_HOST") is None:
        config.load_kube_config()
    else:
        config.load_incluster_config()

    target_node = get_target_node()
    all_metrics = []
    print("=" * 60)
    print("HostPath Capacity Test")
    print("  Node: {}".format(target_node))
    print("  Max datasets: {}".format(MAX_DATASETS))
    print("  Data size per dataset: {}MB".format(DATA_SIZE_MB))
    print("=" * 60)

    failed = False
    for i in range(1, MAX_DATASETS + 1):
        print("\n--- Iteration {}/{}: deploying dataset {} ---".format(i, MAX_DATASETS, i))

        # Collect pre-iteration metrics
        pre_metrics = collect_node_metrics(target_node)
        print("Pre-iteration node metrics: cpu={}, memory={}".format(
            pre_metrics["cpu_usage"], pre_metrics["memory_usage"]
        ))

        start_time = time.time()

        flow = run_capacity_iteration(i, target_node)
        try:
            flow.run()
        except Exception as e:
            print("Iteration {} FAILED: {}".format(i, e))
            failed = True
            # Collect metrics even on failure
            post_metrics = collect_node_metrics(target_node)
            oom_kills = check_for_oom_kills("default")
            iteration_result = {
                "iteration": i,
                "status": "FAILED",
                "error": str(e),
                "pre_metrics": pre_metrics,
                "post_metrics": post_metrics,
                "oom_kills": oom_kills,
            }
            all_metrics.append(iteration_result)
            print("Stopping capacity test at iteration {} due to failure".format(i))
            break

        elapsed = time.time() - start_time
        post_metrics = collect_node_metrics(target_node)
        oom_kills = check_for_oom_kills("default")
        pod_metrics = collect_pod_metrics("default", "app=alluxio")

        iteration_result = {
            "iteration": i,
            "status": "PASSED",
            "elapsed_seconds": round(elapsed, 2),
            "pre_metrics": pre_metrics,
            "post_metrics": post_metrics,
            "oom_kills": oom_kills,
            "pod_metrics": pod_metrics,
        }
        all_metrics.append(iteration_result)

        print("Iteration {} completed in {:.1f}s".format(i, elapsed))
        print("Post-iteration node metrics: cpu={}, memory={}".format(
            post_metrics["cpu_usage"], post_metrics["memory_usage"]
        ))
        if oom_kills:
            print("WARNING: OOMKills detected: {}".format(oom_kills))

    # Print summary
    print("\n" + "=" * 60)
    print("CAPACITY TEST SUMMARY")
    print("=" * 60)
    for result in all_metrics:
        status = result["status"]
        iteration = result["iteration"]
        if status == "PASSED":
            print("  Iteration {}: PASSED in {:.1f}s | cpu={} memory={}".format(
                iteration,
                result["elapsed_seconds"],
                result["post_metrics"]["cpu_usage"],
                result["post_metrics"]["memory_usage"],
            ))
        else:
            print("  Iteration {}: FAILED - {}".format(iteration, result.get("error", "unknown")))

    # Write metrics to file for external consumption
    metrics_file = os.getenv("HOSTPATH_CAPACITY_METRICS_FILE", "/tmp/hostpath_capacity_metrics.json")
    try:
        with open(metrics_file, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print("\nMetrics written to {}".format(metrics_file))
    except Exception as e:
        print("Warning: could not write metrics file: {}".format(e))

    if failed:
        exit(1)


if __name__ == '__main__':
    main()
