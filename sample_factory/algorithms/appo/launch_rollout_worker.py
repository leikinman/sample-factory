import os
import sys
import json
import argparse
import time
import signal
from easydict import EasyDict
import threading

from sample_factory.algorithms.appo.actor_worker import ActorWorker
from sample_factory.algorithms.appo.appo_utils import make_env_func
from sample_factory.algorithms.appo.shared_buffers import SharedBuffers
from sample_factory.utils.utils import log

if os.name == 'nt':
    from sample_factory.utils import Queue as MpQueue
else:
    from faster_fifo import Queue as MpQueue
    # noinspection PyUnresolvedReferences
    import faster_fifo_reduction

stop_event = threading.Event()

def my_handler(signum, frame):
    global stop_event
    stop_event.set()

def main():
    global stop_event

    parser = argparse.ArgumentParser(description=r'Launch a rollout worker process.')

    parser.add_argument('-c', '--config_path', required=True, type=str, help='The config.')
    parser.add_argument('--worker_idx', required=True, type=int, help='The rollout worker index.')

    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        cfg = json.load(f)
    cfg = EasyDict(cfg)

    tmp_env = make_env_func(cfg, env_config=None)

    obs_space = tmp_env.observation_space
    action_space = tmp_env.action_space
    num_agents = tmp_env.num_agents
    idx = args.worker_idx

    traj_buffers = SharedBuffers(cfg, num_agents, obs_space, action_space, create=False)

    actor_queue = MpQueue(name='actor_queue_{}'.format(idx), create=False)
    
    policy_queues = dict()
    learner_queues = []
    for policy_id in range(cfg.num_policies):
        learner_id = policy_id
        policy_queues[policy_id] = MpQueue(name='policy_queue_{}'.format(policy_id), create=False)
        learner_queues.append( MpQueue(name='learner_worker_task_queue_{}'.format(learner_id), create=False) )

    report_queue = MpQueue(name='report_queue', create=False)
    w = ActorWorker(
        cfg, obs_space, action_space, num_agents, idx, traj_buffers,
        task_queue=actor_queue, policy_queues=policy_queues,
        report_queue=report_queue, learner_queues=learner_queues,
    )

    signal.signal(signal.SIGINT, my_handler)
    signal.signal(signal.SIGTERM, my_handler)

    stop_event.wait()
    while True:
        try:
            w.join()
            break
        except Exception as e:
            print(str(e))
            break


if __name__ == '__main__':
    sys.exit(main())
