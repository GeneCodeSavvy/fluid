#!/bin/bash

testname="hostpath accelerate basic e2e"

dataset_name="hostpath-data"
job_name="hostpath-fluid-test"
hostpath_dir="/mnt/data/hostpath-test"
node_label="fluid.io/hostpath-test"

function syslog() {
    echo ">>> $1"
}

function panic() {
    err_msg=$1
    syslog "test \"$testname\" failed: $err_msg"
    exit 1
}

function prepare_hostpath() {
    syslog "Preparing host path directory and test data"

    # Label a node for hostpath test
    node=$(kubectl get nodes --no-headers -o custom-columns=":metadata.name" | head -1)
    if [[ -z "$node" ]]; then
        panic "no nodes available in the cluster"
    fi

    kubectl label node "$node" ${node_label}=true --overwrite
    syslog "Labeled node $node with ${node_label}=true"

    # Create the host directory with sample data on the target node using a DaemonSet
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: hostpath-data-setup
spec:
  nodeName: ${node}
  restartPolicy: Never
  containers:
    - name: setup
      image: busybox
      command: ["/bin/sh"]
      args:
      - -c
      - |
        set -ex
        mkdir -p ${hostpath_dir}
        # Generate synthetic test data: 100 small files (1KB each)
        for i in \$(seq 1 100); do
          dd if=/dev/urandom of=${hostpath_dir}/file_\${i}.dat bs=1024 count=1 2>/dev/null
        done
        # Generate one medium file (10MB)
        dd if=/dev/urandom of=${hostpath_dir}/medium_file.dat bs=1048576 count=10 2>/dev/null
        echo "Data setup complete"
        ls -la ${hostpath_dir}
      volumeMounts:
        - name: host-vol
          mountPath: /mnt/data
  volumes:
    - name: host-vol
      hostPath:
        path: /mnt/data
        type: DirectoryOrCreate
EOF

    # Wait for the setup pod to complete
    deadline=120
    counter=0
    while true; do
        phase=$(kubectl get pod hostpath-data-setup -ojsonpath='{@.status.phase}' 2>/dev/null || echo "Unknown")
        if [[ "$phase" == "Succeeded" ]]; then
            break
        fi
        if [[ "$phase" == "Failed" ]]; then
            kubectl logs hostpath-data-setup
            panic "data setup pod failed"
        fi
        if [[ $counter -ge $deadline ]]; then
            panic "timeout waiting for data setup pod (${deadline}s)"
        fi
        sleep 2
        counter=$((counter + 2))
    done
    syslog "Host path test data prepared at ${hostpath_dir} on node $node"
}

function create_dataset() {
    kubectl create -f test/gha-e2e/hostpath/dataset.yaml

    if [[ -z "$(kubectl get dataset $dataset_name -oname)" ]]; then
        panic "failed to create dataset"
    fi

    if [[ -z "$(kubectl get alluxioruntime $dataset_name -oname)" ]]; then
        panic "failed to create alluxioruntime"
    fi
}

function wait_dataset_bound() {
    deadline=180 # 3 minutes
    last_state=""
    log_interval=0
    log_times=0
    while true; do
        last_state=$(kubectl get dataset $dataset_name -ojsonpath='{@.status.phase}')
        if [[ $log_interval -eq 3 ]]; then
            log_times=$(expr $log_times + 1)
            syslog "checking dataset.status.phase==Bound (already $(expr $log_times \* $log_interval \* 5)s, last state: $last_state)"
            if [[ "$(expr $log_times \* $log_interval \* 5)" -ge "$deadline" ]]; then
                panic "timeout for ${deadline}s!"
            fi
            log_interval=0
        fi

        if [[ "$last_state" == "Bound" ]]; then
            break
        fi
        log_interval=$(expr $log_interval + 1)
        sleep 5
    done
    syslog "Found dataset $dataset_name status.phase==Bound"
}

function create_job() {
    kubectl create -f test/gha-e2e/hostpath/job.yaml

    if [[ -z "$(kubectl get job $job_name -oname)" ]]; then
        panic "failed to create job"
    fi
}

function wait_job_completed() {
    deadline=300 # 5 minutes
    counter=0
    while true; do
        succeed=$(kubectl get job $job_name -ojsonpath='{@.status.succeeded}')
        failed=$(kubectl get job $job_name -ojsonpath='{@.status.failed}')
        if [[ "$failed" -ne "0" ]] 2>/dev/null; then
            syslog "Job logs:"
            kubectl logs job/$job_name || true
            panic "job failed when accessing data"
        fi
        if [[ "$succeed" -eq "1" ]] 2>/dev/null; then
            break
        fi
        if [[ $counter -ge $deadline ]]; then
            panic "timeout waiting for job completion (${deadline}s)"
        fi
        sleep 5
        counter=$((counter + 5))
    done
    syslog "Found succeeded job $job_name"

    # Print job logs for throughput info
    syslog "Job output:"
    kubectl logs job/$job_name || true
}

function dump_env_and_clean_up() {
    syslog "Collecting diagnostics"
    kubectl get dataset $dataset_name -o yaml || true
    kubectl get alluxioruntime $dataset_name -o yaml || true
    kubectl get pods -l app=alluxio -o wide || true

    syslog "Cleaning up resources for testcase $testname"
    kubectl delete -f test/gha-e2e/hostpath/ --ignore-not-found
    kubectl delete pod hostpath-data-setup --ignore-not-found

    # Remove node label
    node=$(kubectl get nodes -l ${node_label}=true --no-headers -o custom-columns=":metadata.name" | head -1)
    if [[ -n "$node" ]]; then
        kubectl label node "$node" ${node_label}- || true
    fi
}

function main() {
    syslog "[TESTCASE $testname STARTS AT $(date)]"
    prepare_hostpath
    trap dump_env_and_clean_up EXIT
    create_dataset
    wait_dataset_bound
    create_job
    wait_job_completed
    syslog "[TESTCASE $testname SUCCEEDED AT $(date)]"
}

main
