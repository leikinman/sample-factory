import os
import sys
import copy
import time
import argparse
import multiprocessing
import signal
import importlib

from kubernetes import client, config

from sample_factory.utils.utils import log
from sample_factory.algorithms.utils.algo_utils import ExperimentStatus

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=None, type=str, help='Name of the python module that describes the run, e.g. sample_factory.runner.runs.doom_battle_hybrid')
    parser.add_argument('--namespace', default="default", type=str, help='k8s namespace.')
    subparsers = parser.add_subparsers()

    # Start
    parser_start = subparsers.add_parser("start", help="start a new experiment.")
    parser_start.add_argument('--work_dir', required=True, help="The shared path between different worker.")
    parser_start.add_argument('--node', default=None, type=str, help="the node to execute. Recommended to check whether the resources of the node are sufficient first.")
    parser_start.add_argument('--cpus_per_actor', default=1, type=int, help="CPUs in a single actor.")
    parser_start.add_argument('--cpus_main_proc', default=2, type=int, help="CPUs in a main procress.")
    parser_start.add_argument('--gpus_main_proc', default=1, type=int, help="GPUs in a main procress.")
    parser_start.set_defaults(func=start_exp)

    # Increate Rollout worker
    parser_increase = subparsers.add_parser("increase", help="increase rollout worker.")
    parser_increase.set_defaults(func=increase_rollout_worker)

    # Decreate Rollout worker
    parser_increase = subparsers.add_parser("decrease", help="decrease rollout worker.")
    parser_increase.set_defaults(func=decrease_rollout_worker)

    # Stop
    parser_stop = subparsers.add_parser("stop", help="stop experiment.")
    parser_stop.set_defaults(func=stop_exp)

    return parser

TEMP = {
    'apiVersion': 'v1', 
    'kind': 'Pod', 
    'metadata': {
        'name': None,
        }, 
    'spec': {
        'restartPolicy': 'Never',
        # 'nodeName': None,
        'hostIPC': True, 
        'volumes': [
            {'name': 'data', 'hostPath': {'path': None}}, 
            {'name': 'cache-volume', 'hostPath': {'path': '/dev/shm'}}
        ], 
        'containers': [
            {
                'name': 'sample-factory', 
                'image': 'registry.sensetime.com/cloudnative4ai/sample-factory:v1.120.2-ljw', 
                'imagePullPolicy': 'Always', 
                'resources': {
                    'limits': {
                        # resource request
                        }
                },
                'command': ['/bin/bash', '-c'], 
                'args': None, 
                'volumeMounts': [
                    {'name': 'data', 'mountPath': '/lustre'}, 
                    {'name': 'cache-volume', 'mountPath': '/dev/shm'}
                ]
            }
        ]
    }
}
ROLLOUT_WORKER_CMD = "python -m sample_factory.algorithms.appo.launch_rollout_worker -c=/lustre/{} --worker_idx={}"


def _find_main_proc_pod(pod_list, experiment_name, verbose=True):
    main_proc_pod_name = "{}-main".format(experiment_name)
    pod_names = [pod.metadata.name for pod in pod_list.items]
    if main_proc_pod_name in pod_names:
        idx = pod_names.index(main_proc_pod_name)
        if pod_list.items[idx].status.phase != "Running":
            if verbose:
                log.warning("Please confirm if the experiment {} is running.".format(main_proc_pod_name))
            return None
    else:
        if verbose:
            log.warning("Please confirm the experiment {} exists.".format(main_proc_pod_name))
        return None
    return pod_list.items[idx]

def _get_max_rollout_worker_index_pod(pod_list, experiment_name):
    max_index_pod = None
    worker_idx = -1
    for pod in pod_list.items:
        if "{}-rollout-worker-".format(experiment_name) in pod.metadata.name:
            idx = int(pod.metadata.name.split('-')[-1])
            if idx > worker_idx:
                worker_idx = idx
                max_index_pod = pod
    return max_index_pod, worker_idx

def start_exp(run_description, args):
    config.load_kube_config()
    k8s_core_v1 = client.CoreV1Api()

    experiments = run_description.experiments
    # only have one exp in k8s now
    experiment = experiments[0]
    log.info('Starting processes with cmd: %r', experiment.cmd)

    cmd = experiment.cmd
    name = experiment.base_name

    # Get number worker from cmd
    num_workers = None
    for arg in cmd.split():
        if '--num_workers=' in arg:
            num_workers = int(arg.split("=")[-1])
    if num_workers == None:
        log.error('Cannot find num_workers in cmd')
        return

    body = copy.deepcopy(TEMP)
    body['metadata']['name'] = "{}-main".format(name)
    body['spec']['containers'][0]['args'] = [cmd]
    if args.node:
        body['spec']['nodeName'] = args.node
    body['spec']['volumes'][0]['hostPath']['path'] = args.work_dir
    body['spec']['containers'][0]['resources']['limits']['cpu'] = args.cpus_main_proc
    body['spec']['containers'][0]['resources']['limits']['nvidia.com/gpu'] = args.gpus_main_proc

    log.info('Create experiment {}..'.format(name))
    log.info('Create Main Procress. Pod name is {} ..'.format(body['metadata']['name']))
    main_pod_info = k8s_core_v1.create_namespaced_pod(body=body, namespace=args.namespace)
    sched_node_name = main_pod_info.spec.node_name
    log.info('Main Procress is scheduled to {}, so the rest of the workers will be scheduled there ..'.format(sched_node_name))

    _max_retry = 60
    _t = 0
    while True:
        _t += 1
        pod_list = k8s_core_v1.list_namespaced_pod(namespace=args.namespace)
        pod = _find_main_proc_pod(pod_list=pod_list, experiment_name=name, verbose=False)
        if pod is not None:
            break
        elif _t == _max_retry:
            log.error('There seems to be something wrong with the main process...?')
            return
        else:
            log.info('Sleep 1 sec and wait for the main process to be ready ..')
            time.sleep(1)
                
    for idx in range(num_workers):
        body = copy.deepcopy(TEMP)
        cmd = ROLLOUT_WORKER_CMD.format("{}/cfg.json".format(name), idx)
        body['metadata']['name'] = "{}-rollout-worker-{}".format(name, idx)
        body['spec']['containers'][0]['args'] = [cmd]
        body['spec']['nodeName'] = sched_node_name
        body['spec']['volumes'][0]['hostPath']['path'] = args.work_dir
        body['spec']['containers'][0]['resources']['limits']['cpu'] = args.cpus_per_actor
        log.info('Create Rollout Worker {}. Pod name is {} ..'.format(idx, body['metadata']['name']))
        k8s_core_v1.create_namespaced_pod(body=body, namespace=args.namespace)

def increase_rollout_worker(run_description, args):
    config.load_kube_config()
    k8s_core_v1 = client.CoreV1Api()

    experiments = run_description.experiments
    # only have one exp in k8s now
    experiment = experiments[0]
    name = experiment.base_name

    # The pod list in namespace
    pod_list = k8s_core_v1.list_namespaced_pod(namespace=args.namespace)

    if _find_main_proc_pod(pod_list=pod_list, experiment_name=name) is None:
        return
    
    max_index_pod, worker_idx = _get_max_rollout_worker_index_pod(pod_list=pod_list, experiment_name=name)
    cmd = max_index_pod.spec.containers[0].args[-1]

    # increase index and update cmd
    worker_idx += 1
    cmd_split = cmd.split()
    cmd_split[-1] = cmd_split[-1].split("=")[0] + "=" + str(worker_idx)
    cmd = " ".join(cmd_split)

    # Send a SIGUSR1 signal to main proc that it needs to added a new worker.
    main_proc_pod_name = "{}-main".format(name)
    os.system(
        "kubectl exec -it {} -n {} -- bash -c 'kill -{} 1'".format(main_proc_pod_name, args.namespace, int(signal.SIGUSR1))
        )

    body = copy.deepcopy(TEMP)
    body['metadata']['name'] = "{}-rollout-worker-{}".format(name, worker_idx)
    body['spec'] = max_index_pod.spec
    body['spec'].containers[0].args = [cmd]
    log.info('Create Rollout Worker {}. Pod name is {} ..'.format(worker_idx, body['metadata']['name']))
    k8s_core_v1.create_namespaced_pod(body=body, namespace=args.namespace)

def decrease_rollout_worker(run_description, args):
    config.load_kube_config()
    k8s_core_v1 = client.CoreV1Api()

    experiments = run_description.experiments
    # only have one exp in k8s now
    experiment = experiments[0]
    name = experiment.base_name

    # The pod list in namespace
    pod_list = k8s_core_v1.list_namespaced_pod(namespace=args.namespace)

    if _find_main_proc_pod(pod_list=pod_list, experiment_name=name) is None:
        return

    max_index_pod, _ = _get_max_rollout_worker_index_pod(pod_list=pod_list, experiment_name=name)

    # Send a SIGUSR1 signal to main proc that it needs to added a new worker.
    main_proc_pod_name = "{}-main".format(name)
    os.system(
        "kubectl exec -it {} -n {} -- bash -c 'kill -{} 1'".format(main_proc_pod_name, args.namespace, int(signal.SIGUSR2))
        )

    try:
        pod_name = max_index_pod.metadata.name
        api_response = k8s_core_v1.delete_namespaced_pod(pod_name, args.namespace)
        log.info("pod {} deleted".format(pod_name))
    except client.rest.ApiException as e:
        log.error("Exception when calling CoreV1Api.delete_namespaced_pod: %s\n" % e)

def stop_exp(run_description, args):
    config.load_kube_config()
    k8s_core_v1 = client.CoreV1Api()

    experiments = run_description.experiments
    # only have one exp in k8s now
    experiment = experiments[0]

    cmd = experiment.cmd
    name = experiment.base_name

    pod_list = k8s_core_v1.list_namespaced_pod(namespace=args.namespace)
    pod_names = [pod.metadata.name for pod in pod_list.items]
    for pod_name in pod_names:
        if name in pod_name:
            try:
                api_response = k8s_core_v1.delete_namespaced_pod(pod_name, args.namespace)
                log.info("pod {} deleted".format(pod_name))
            except client.rest.ApiException as e:
                log.error("Exception when calling CoreV1Api.delete_namespaced_pod: %s\n" % e)

def main():
    parser = parse_args()
    args = parser.parse_args()

    try:
        # assuming we're given the full name of the module
        run_module = importlib.import_module(f'{args.run}')
    except ImportError:
        try:
            run_module = importlib.import_module(f'sample_factory.runner.runs.{args.run}')
        except ImportError:
            log.error('Could not import the run module')
            return ExperimentStatus.FAILURE

    run_description = run_module.RUN_DESCRIPTION
    # run_description.experiment_suffix = args.experiment_suffix

    args.func(run_description, args)


if __name__ == '__main__':
    sys.exit(main())