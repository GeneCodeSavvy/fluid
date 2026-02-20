"""
TestCase: HostPath multi-user hot and cold data scenarios
DDC Engine: Alluxio

Scenarios:
  1. Cold multi-user: N jobs start simultaneously, all read uncached data.
     Stresses backend (host path) read path and cache fill under contention.
  2. Hot multi-user: DataLoad warms cache first, then N jobs read cached data.
     Stresses cache hit path and fuse/client concurrency.
  3. Mixed hot/cold: Warm cache with subset A, then run jobs on both subset A (hot)
     and subset B (cold). Tests fairness and mixed cache behavior.

Environment variables:
  HOSTPATH_MULTIUSER_NUM_JOBS: Number of concurrent jobs per scenario (default: 4)
  HOSTPATH_MULTIUSER_DATA_SIZE_MB: Size of test data in MB (default: 200)
  HOSTPATH_MULTIUSER_SCENARIO: Which scenario to run (cold|hot|mixed|all, default: all)
  HOSTPATH_MULTIUSER_NODE_LABEL_KEY: Node label key (default: fluid.io/hostpath-test)
  HOSTPATH_MULTIUSER_NODE_LABEL_VALUE: Node label value (default: true)
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
from framework.step import (
    SimpleStep,
    StatusCheckStep,
    dummy_back,
    currying_fn,
)
from framework.exception import TestError

from kubernetes import client, config


def get_env_int(key, default):
    val = os.getenv(key)
    return int(val) if val else default


NUM_JOBS = get_env_int("HOSTPATH_MULTIUSER_NUM_JOBS", 4)
DATA_SIZE_MB = get_env_int("HOSTPATH_MULTIUSER_DATA_SIZE_MB", 200)
SCENARIO = os.getenv("HOSTPATH_MULTIUSER_SCENARIO", "all")
NODE_LABEL_KEY = os.getenv(
    "HOSTPATH_MULTIUSER_NODE_LABEL_KEY", "fluid.io/hostpath-test"
)
NODE_LABEL_VALUE = os.getenv("HOSTPATH_MULTIUSER_NODE_LABEL_VALUE", "true")
HOSTPATH_DIR = "/mnt/data/hostpath-multiuser"
NAMESPACE = "default"


def get_target_node():
    """Get or label a target node."""
    api = client.CoreV1Api()

    nodes = api.list_node(
        label_selector="{}={}".format(NODE_LABEL_KEY, NODE_LABEL_VALUE)
    )
    if nodes.items:
        return nodes.items[0].metadata.name

    all_nodes = api.list_node()
    if not all_nodes.items:
        raise TestError("No nodes available in cluster")

    node_name = all_nodes.items[0].metadata.name
    api.patch_node(
        name=node_name,
        body={"metadata": {"labels": {NODE_LABEL_KEY: NODE_LABEL_VALUE}}},
    )
    print(
        "Labeled node '{}' with {}={}".format(
            node_name, NODE_LABEL_KEY, NODE_LABEL_VALUE
        )
    )
    return node_name


def create_multiuser_setup_pod(target_node, scenario):
    """Create synthetic data on host for multiuser test."""
    api = client.CoreV1Api()
    pod_name = "mu-setup-{}".format(scenario)

    if scenario == "mixed":
        # Create two subsets: subset_a and subset_b
        setup_script = (
            "set -e; "
            "mkdir -p /hostdata/subset_a /hostdata/subset_b; "
            "echo 'Generating subset_a ({}MB)...'; "
            "dd if=/dev/urandom of=/hostdata/subset_a/data.bin bs=1048576 count={} 2>/dev/null; "
            "echo 'Generating subset_b ({}MB)...'; "
            "dd if=/dev/urandom of=/hostdata/subset_b/data.bin bs=1048576 count={} 2>/dev/null; "
            "echo 'Done'; du -sh /hostdata/*"
        ).format(
            DATA_SIZE_MB // 2, DATA_SIZE_MB // 2, DATA_SIZE_MB // 2, DATA_SIZE_MB // 2
        )
    else:
        setup_script = (
            "set -e; "
            "mkdir -p /hostdata; "
            "echo 'Generating {}MB of test data...'; "
            "dd if=/dev/urandom of=/hostdata/data.bin bs=1048576 count={} 2>/dev/null; "
            "echo 'Done'; du -sh /hostdata"
        ).format(DATA_SIZE_MB, DATA_SIZE_MB)

    hostpath_dir = "{}/{}".format(HOSTPATH_DIR, scenario)

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
                    ],
                )
            ],
            volumes=[
                client.V1Volume(
                    name="host-vol",
                    host_path=client.V1HostPathVolumeSource(
                        path=hostpath_dir, type="DirectoryOrCreate"
                    ),
                )
            ],
        ),
    )

    api.create_namespaced_pod(namespace=NAMESPACE, body=pod)
    print("Setup pod '{}' created on node '{}'".format(pod_name, target_node))


def check_setup_pod_fn(scenario):
    def check():
        api = client.CoreV1Api()
        pod_name = "mu-setup-{}".format(scenario)
        try:
            pod = api.read_namespaced_pod(name=pod_name, namespace=NAMESPACE)
            if pod.status.phase == "Succeeded":
                return True
            if pod.status.phase == "Failed":
                raise TestError("Setup pod {} failed".format(pod_name))
        except client.exceptions.ApiException as e:
            if e.status == 404:
                raise TestError("Setup pod {} not found".format(pod_name))
        return False

    return check


def delete_setup_pod_fn(scenario):
    def delete():
        api = client.CoreV1Api()
        try:
            api.delete_namespaced_pod(
                name="mu-setup-{}".format(scenario),
                namespace=NAMESPACE,
                body=client.V1DeleteOptions(propagation_policy="Background"),
            )
        except client.exceptions.ApiException as e:
            if e.status != 404:
                raise

    return delete


def create_concurrent_jobs_fn(ds_name, scenario, num_jobs, script):
    """Create N concurrent jobs that all read from the same dataset."""

    def create():
        api = client.BatchV1Api()

        for i in range(num_jobs):
            job_name = "mu-{}-job-{}".format(scenario, i)
            job_script = (
                "set -ex; "
                "echo '=== Job {} of {} (scenario: {}) ==='; "
                "start=$(date +%s); "
                "{} "
                "end=$(date +%s); "
                'echo "METRIC job_{}_seconds=$((end - start))"'
            ).format(i, num_jobs, scenario, script, i)

            container = client.V1Container(
                name="worker",
                image="busybox",
                command=["/bin/sh"],
                args=["-c", job_script],
                volume_mounts=[
                    client.V1VolumeMount(mount_path="/data", name="data-vol")
                ],
            )

            template = client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels={"app": "mu-{}".format(scenario)}),
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    containers=[container],
                    volumes=[
                        client.V1Volume(
                            name="data-vol",
                            persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                                claim_name=ds_name
                            ),
                        )
                    ],
                ),
            )

            job = client.V1Job(
                api_version="batch/v1",
                kind="Job",
                metadata=client.V1ObjectMeta(name=job_name),
                spec=client.V1JobSpec(template=template, backoff_limit=2),
            )

            api.create_namespaced_job(namespace=NAMESPACE, body=job)
            print("Job '{}' created".format(job_name))

    return create


def check_all_jobs_completed_fn(scenario, num_jobs):
    """Check that all concurrent jobs have completed successfully."""

    def check():
        api = client.BatchV1Api()
        completed = 0
        for i in range(num_jobs):
            job_name = "mu-{}-job-{}".format(scenario, i)
            try:
                response = api.read_namespaced_job_status(
                    name=job_name, namespace=NAMESPACE
                )
                if response.status.failed and response.status.failed > 0:
                    raise TestError("Job '{}' failed".format(job_name))
                if response.status.succeeded and response.status.succeeded >= 1:
                    completed += 1
            except client.exceptions.ApiException as e:
                if e.status == 404:
                    return False
                raise

        return completed == num_jobs

    return check


def delete_all_jobs_fn(scenario, num_jobs):
    """Delete all concurrent jobs for a scenario."""

    def delete():
        api = client.BatchV1Api()
        for i in range(num_jobs):
            job_name = "mu-{}-job-{}".format(scenario, i)
            try:
                api.delete_namespaced_job(
                    name=job_name,
                    namespace=NAMESPACE,
                    body=client.V1DeleteOptions(propagation_policy="Background"),
                )
            except client.exceptions.ApiException as e:
                if e.status != 404:
                    print("Warning: failed to delete job {}: {}".format(job_name, e))

    return delete


def collect_job_metrics(scenario, num_jobs):
    """Collect logs from all jobs to extract METRIC lines."""
    api = client.BatchV1Api()
    core_api = client.CoreV1Api()
    metrics = {}

    for i in range(num_jobs):
        job_name = "mu-{}-job-{}".format(scenario, i)
        try:
            pods = core_api.list_namespaced_pod(
                namespace=NAMESPACE, label_selector="job-name={}".format(job_name)
            )
            for pod in pods.items:
                try:
                    log = core_api.read_namespaced_pod_log(
                        name=pod.metadata.name, namespace=NAMESPACE
                    )
                    for line in log.split("\n"):
                        if line.startswith("METRIC "):
                            key_val = line[7:].strip()
                            if "=" in key_val:
                                k, v = key_val.split("=", 1)
                                metrics["{}_{}".format(job_name, k)] = v
                except Exception:
                    pass
        except Exception:
            pass

    return metrics


def print_job_logs(scenario, num_jobs):
    """Print logs from all jobs for debugging."""
    core_api = client.CoreV1Api()
    for i in range(num_jobs):
        job_name = "mu-{}-job-{}".format(scenario, i)
        try:
            pods = core_api.list_namespaced_pod(
                namespace=NAMESPACE, label_selector="job-name={}".format(job_name)
            )
            for pod in pods.items:
                try:
                    log = core_api.read_namespaced_pod_log(
                        name=pod.metadata.name, namespace=NAMESPACE, tail_lines=20
                    )
                    print("--- Logs for {} ---".format(job_name))
                    print(log)
                except Exception:
                    pass
        except Exception:
            pass


def run_cold_scenario(target_node):
    """Cold: N jobs start simultaneously, all read uncached data."""
    scenario = "cold"
    ds_name = "mu-cold"
    hostpath_dir = "{}/{}".format(HOSTPATH_DIR, scenario)

    mount = fluidapi.Mount()
    mount.set_mount_info(ds_name, "local://{}".format(hostpath_dir))

    dataset = fluidapi.Dataset(ds_name, NAMESPACE)
    dataset.add_mount(mount.dump())
    dataset.set_node_affinity(NODE_LABEL_KEY, NODE_LABEL_VALUE)

    runtime = fluidapi.Runtime("AlluxioRuntime", ds_name, NAMESPACE)
    runtime.set_replicas(1)
    runtime.set_tieredstore("MEM", "/dev/shm", "{}Mi".format(DATA_SIZE_MB * 2))

    read_script = "cat /data/data.bin > /dev/null; "

    flow = TestFlow(
        "Multi-user Cold - {} concurrent jobs on uncached data".format(NUM_JOBS)
    )

    # Setup data
    flow.append_step(
        SimpleStep(
            step_name="setup data for cold scenario",
            forth_fn=currying_fn(
                create_multiuser_setup_pod, target_node=target_node, scenario=scenario
            ),
            back_fn=delete_setup_pod_fn(scenario),
        )
    )
    flow.append_step(
        StatusCheckStep(
            step_name="wait for cold setup",
            forth_fn=check_setup_pod_fn(scenario),
            timeout=300,
            interval=5,
        )
    )

    # Create Dataset + Runtime
    flow.append_step(
        SimpleStep(
            step_name="create cold dataset",
            forth_fn=funcs.create_dataset_fn(dataset.dump()),
            back_fn=funcs.delete_dataset_and_runtime_fn(
                runtime.dump(), ds_name, NAMESPACE
            ),
        )
    )
    flow.append_step(
        SimpleStep(
            step_name="create cold runtime",
            forth_fn=funcs.create_runtime_fn(runtime.dump()),
            back_fn=dummy_back,
        )
    )
    flow.append_step(
        StatusCheckStep(
            step_name="check cold dataset bound",
            forth_fn=funcs.check_dataset_bound_fn(ds_name, NAMESPACE),
            timeout=300,
        )
    )

    # Launch all jobs simultaneously (cold - no prior cache)
    flow.append_step(
        SimpleStep(
            step_name="create {} concurrent cold read jobs".format(NUM_JOBS),
            forth_fn=create_concurrent_jobs_fn(
                ds_name, scenario, NUM_JOBS, read_script
            ),
            back_fn=delete_all_jobs_fn(scenario, NUM_JOBS),
        )
    )
    flow.append_step(
        StatusCheckStep(
            step_name="check all cold jobs completed",
            forth_fn=check_all_jobs_completed_fn(scenario, NUM_JOBS),
            timeout=900,
            interval=10,
        )
    )

    return flow


def run_hot_scenario(target_node):
    """Hot: DataLoad warms cache, then N jobs read cached data."""
    scenario = "hot"
    ds_name = "mu-hot"
    hostpath_dir = "{}/{}".format(HOSTPATH_DIR, scenario)
    dataload_name = "mu-hot-warmup"

    mount = fluidapi.Mount()
    mount.set_mount_info(ds_name, "local://{}".format(hostpath_dir))

    dataset = fluidapi.Dataset(ds_name, NAMESPACE)
    dataset.add_mount(mount.dump())
    dataset.set_node_affinity(NODE_LABEL_KEY, NODE_LABEL_VALUE)

    runtime = fluidapi.Runtime("AlluxioRuntime", ds_name, NAMESPACE)
    runtime.set_replicas(1)
    runtime.set_tieredstore("MEM", "/dev/shm", "{}Mi".format(DATA_SIZE_MB * 2))

    dataload = fluidapi.DataLoad(dataload_name, NAMESPACE)
    dataload.set_target_dataset(ds_name, NAMESPACE)
    dataload.set_load_metadata(True)

    read_script = "cat /data/data.bin > /dev/null; "

    flow = TestFlow(
        "Multi-user Hot - {} concurrent jobs on warm cache".format(NUM_JOBS)
    )

    # Setup data
    flow.append_step(
        SimpleStep(
            step_name="setup data for hot scenario",
            forth_fn=currying_fn(
                create_multiuser_setup_pod, target_node=target_node, scenario=scenario
            ),
            back_fn=delete_setup_pod_fn(scenario),
        )
    )
    flow.append_step(
        StatusCheckStep(
            step_name="wait for hot setup",
            forth_fn=check_setup_pod_fn(scenario),
            timeout=300,
            interval=5,
        )
    )

    # Create Dataset + Runtime
    flow.append_step(
        SimpleStep(
            step_name="create hot dataset",
            forth_fn=funcs.create_dataset_fn(dataset.dump()),
            back_fn=funcs.delete_dataset_and_runtime_fn(
                runtime.dump(), ds_name, NAMESPACE
            ),
        )
    )
    flow.append_step(
        SimpleStep(
            step_name="create hot runtime",
            forth_fn=funcs.create_runtime_fn(runtime.dump()),
            back_fn=dummy_back,
        )
    )
    flow.append_step(
        StatusCheckStep(
            step_name="check hot dataset bound",
            forth_fn=funcs.check_dataset_bound_fn(ds_name, NAMESPACE),
            timeout=300,
        )
    )

    # Warm cache via DataLoad
    flow.append_step(
        SimpleStep(
            step_name="create DataLoad to warm cache",
            forth_fn=funcs.create_dataload_fn(dataload.dump()),
            back_fn=dummy_back,
        )
    )
    flow.append_step(
        StatusCheckStep(
            step_name="check DataLoad completed (cache warm)",
            forth_fn=funcs.check_dataload_job_status_fn(dataload_name, NAMESPACE),
            timeout=600,
            interval=10,
        )
    )

    # Verify cache is populated
    flow.append_step(
        StatusCheckStep(
            step_name="check dataset is fully cached",
            forth_fn=funcs.check_dataset_cached_percentage_fn(ds_name, NAMESPACE),
            timeout=120,
            interval=5,
        )
    )

    # Launch all jobs on hot (cached) data
    flow.append_step(
        SimpleStep(
            step_name="create {} concurrent hot read jobs".format(NUM_JOBS),
            forth_fn=create_concurrent_jobs_fn(
                ds_name, scenario, NUM_JOBS, read_script
            ),
            back_fn=delete_all_jobs_fn(scenario, NUM_JOBS),
        )
    )
    flow.append_step(
        StatusCheckStep(
            step_name="check all hot jobs completed",
            forth_fn=check_all_jobs_completed_fn(scenario, NUM_JOBS),
            timeout=600,
            interval=10,
        )
    )

    return flow


def run_mixed_scenario(target_node):
    """Mixed: Warm cache with subset_a, then run jobs on both subset_a (hot) and subset_b (cold)."""
    scenario = "mixed"
    ds_name = "mu-mixed"
    hostpath_dir = "{}/{}".format(HOSTPATH_DIR, scenario)
    dataload_name = "mu-mixed-warmup"

    mount = fluidapi.Mount()
    mount.set_mount_info(ds_name, "local://{}".format(hostpath_dir))

    dataset = fluidapi.Dataset(ds_name, NAMESPACE)
    dataset.add_mount(mount.dump())
    dataset.set_node_affinity(NODE_LABEL_KEY, NODE_LABEL_VALUE)

    runtime = fluidapi.Runtime("AlluxioRuntime", ds_name, NAMESPACE)
    runtime.set_replicas(1)
    runtime.set_tieredstore("MEM", "/dev/shm", "{}Mi".format(DATA_SIZE_MB * 2))

    # DataLoad only subset_a (warm it)
    dataload = fluidapi.DataLoad(dataload_name, NAMESPACE)
    dataload.set_target_dataset(ds_name, NAMESPACE)
    dataload.set_load_metadata(True)

    # Half jobs read subset_a (hot), half read subset_b (cold)
    hot_script = "cat /data/subset_a/data.bin > /dev/null; "
    cold_script = "cat /data/subset_b/data.bin > /dev/null; "

    flow = TestFlow(
        "Multi-user Mixed - {} hot and cold jobs on same dataset".format(NUM_JOBS)
    )

    # Setup data (two subsets)
    flow.append_step(
        SimpleStep(
            step_name="setup data for mixed scenario",
            forth_fn=currying_fn(
                create_multiuser_setup_pod, target_node=target_node, scenario=scenario
            ),
            back_fn=delete_setup_pod_fn(scenario),
        )
    )
    flow.append_step(
        StatusCheckStep(
            step_name="wait for mixed setup",
            forth_fn=check_setup_pod_fn(scenario),
            timeout=300,
            interval=5,
        )
    )

    # Create Dataset + Runtime
    flow.append_step(
        SimpleStep(
            step_name="create mixed dataset",
            forth_fn=funcs.create_dataset_fn(dataset.dump()),
            back_fn=funcs.delete_dataset_and_runtime_fn(
                runtime.dump(), ds_name, NAMESPACE
            ),
        )
    )
    flow.append_step(
        SimpleStep(
            step_name="create mixed runtime",
            forth_fn=funcs.create_runtime_fn(runtime.dump()),
            back_fn=dummy_back,
        )
    )
    flow.append_step(
        StatusCheckStep(
            step_name="check mixed dataset bound",
            forth_fn=funcs.check_dataset_bound_fn(ds_name, NAMESPACE),
            timeout=300,
        )
    )

    # Warm subset_a via DataLoad (this warms the entire dataset; subset_a will be hot from first read)
    # For a more targeted warm, run a job that reads only subset_a first
    warm_script = "cat /data/subset_a/data.bin > /dev/null; echo 'Warmup complete'"
    warm_job_name = "mu-mixed-warmup-job"
    flow.append_step(
        SimpleStep(
            step_name="warm subset_a via read job",
            forth_fn=funcs.create_job_fn(
                warm_script, ds_name, name=warm_job_name, namespace=NAMESPACE
            ),
            back_fn=funcs.delete_job_fn(name=warm_job_name, namespace=NAMESPACE),
        )
    )
    flow.append_step(
        StatusCheckStep(
            step_name="check warmup job completed",
            forth_fn=funcs.check_job_status_fn(name=warm_job_name, namespace=NAMESPACE),
            timeout=300,
            interval=5,
        )
    )

    # Launch hot jobs (subset_a) and cold jobs (subset_b) concurrently
    hot_count = NUM_JOBS // 2
    cold_count = NUM_JOBS - hot_count

    flow.append_step(
        SimpleStep(
            step_name="create {} hot jobs on subset_a".format(hot_count),
            forth_fn=create_concurrent_jobs_fn(
                ds_name, "mixed-hot", hot_count, hot_script
            ),
            back_fn=delete_all_jobs_fn("mixed-hot", hot_count),
        )
    )

    flow.append_step(
        SimpleStep(
            step_name="create {} cold jobs on subset_b".format(cold_count),
            forth_fn=create_concurrent_jobs_fn(
                ds_name, "mixed-cold", cold_count, cold_script
            ),
            back_fn=delete_all_jobs_fn("mixed-cold", cold_count),
        )
    )

    flow.append_step(
        StatusCheckStep(
            step_name="check all hot jobs completed",
            forth_fn=check_all_jobs_completed_fn("mixed-hot", hot_count),
            timeout=600,
            interval=10,
        )
    )

    flow.append_step(
        StatusCheckStep(
            step_name="check all cold jobs completed",
            forth_fn=check_all_jobs_completed_fn("mixed-cold", cold_count),
            timeout=600,
            interval=10,
        )
    )

    return flow


def main():
    if os.getenv("KUBERNETES_SERVICE_HOST") is None:
        config.load_kube_config()
    else:
        config.load_incluster_config()

    target_node = get_target_node()

    scenarios = {
        "cold": run_cold_scenario,
        "hot": run_hot_scenario,
        "mixed": run_mixed_scenario,
    }

    if SCENARIO == "all":
        to_run = scenarios
    elif SCENARIO in scenarios:
        to_run = {SCENARIO: scenarios[SCENARIO]}
    else:
        print(
            "Unknown scenario '{}'. Available: {}".format(
                SCENARIO, ", ".join(scenarios.keys())
            )
        )
        exit(1)

    print("=" * 60)
    print("HostPath Multi-User Hot/Cold Test")
    print("  Node: {}".format(target_node))
    print("  Concurrent jobs per scenario: {}".format(NUM_JOBS))
    print("  Data size: {}MB".format(DATA_SIZE_MB))
    print("  Scenarios: {}".format(", ".join(to_run.keys())))
    print("=" * 60)

    results = {}
    failed = False

    for scenario_name, scenario_fn in to_run.items():
        print("\n" + "=" * 60)
        print("Running scenario: {}".format(scenario_name))
        print("=" * 60 + "\n")

        start_time = time.time()
        flow = scenario_fn(target_node)

        try:
            flow.run()
            elapsed = time.time() - start_time
            results[scenario_name] = {
                "status": "PASSED",
                "elapsed_seconds": round(elapsed, 2),
            }

            # Collect metrics from job logs
            if scenario_name == "mixed":
                hot_count = NUM_JOBS // 2
                cold_count = NUM_JOBS - hot_count
                hot_metrics = collect_job_metrics("mixed-hot", hot_count)
                cold_metrics = collect_job_metrics("mixed-cold", cold_count)
                results[scenario_name]["hot_metrics"] = hot_metrics
                results[scenario_name]["cold_metrics"] = cold_metrics
                print_job_logs("mixed-hot", hot_count)
                print_job_logs("mixed-cold", cold_count)
            else:
                metrics = collect_job_metrics(scenario_name, NUM_JOBS)
                results[scenario_name]["metrics"] = metrics
                print_job_logs(scenario_name, NUM_JOBS)

        except Exception as e:
            elapsed = time.time() - start_time
            results[scenario_name] = {
                "status": "FAILED",
                "elapsed_seconds": round(elapsed, 2),
                "error": str(e),
            }
            print("Scenario '{}' FAILED: {}".format(scenario_name, e))
            failed = True

    # Print summary
    print("\n" + "=" * 60)
    print("MULTI-USER TEST SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        status = result["status"]
        elapsed = result["elapsed_seconds"]
        if status == "PASSED":
            print("  {}: PASSED in {:.1f}s".format(name, elapsed))
        else:
            print(
                "  {}: FAILED in {:.1f}s - {}".format(
                    name, elapsed, result.get("error", "")
                )
            )

    # Write results to file
    results_file = os.getenv(
        "HOSTPATH_MULTIUSER_RESULTS_FILE", "/tmp/hostpath_multiuser_results.json"
    )
    try:
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print("\nResults written to {}".format(results_file))
    except Exception as e:
        print("Warning: could not write results file: {}".format(e))

    if failed:
        exit(1)


if __name__ == "__main__":
    main()
