import time
import subprocess
from sample_factory.utils.utils import log
from sample_factory.algorithms.appo.appo_utils import TaskType
import sys

class FakeActorWorker:

    def __init__(
        self, cfg, obs_space, action_space, num_agents, worker_idx, shared_buffers,
        task_queue, policy_queues, report_queue, learner_queues,
    ):
        self.cfg = cfg
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_agents = num_agents

        self.worker_idx = worker_idx

        self.shared_buffers = shared_buffers

        self.terminate = False

        self.num_complete_rollouts = 0

        self.vector_size = cfg.num_envs_per_worker
        self.num_splits = cfg.worker_num_splits
        assert self.vector_size >= self.num_splits
        assert self.vector_size % self.num_splits == 0, 'Vector size should be divisible by num_splits'

        self.env_runners = None

        self.policy_queues = policy_queues
        self.report_queue = report_queue
        self.learner_queues = learner_queues
        self.task_queue = task_queue

        self.reward_shaping = [None for _ in range(self.cfg.num_policies)]

        exec_cmd = [sys.executable,
                    '-m', 'sample_factory.algorithms.appo.launch_rollout_worker',
                    '-c', '{}/{}/cfg.json'.format(cfg.train_dir, cfg.experiment),
                    '--worker_id', '{}'.format(self.worker_idx)
                    ]
        self.process = subprocess.Popen(exec_cmd)

    def init(self):
        self.task_queue.put((TaskType.INIT, None))

    def request_reset(self):
        self.task_queue.put((TaskType.RESET, None))

    def request_step(self, split, actions):
        data = (split, actions)
        self.task_queue.put((TaskType.ROLLOUT_STEP, data))

    def close(self):
        self.task_queue.put((TaskType.TERMINATE, None))
        # TODO: For Test
        self.process.terminate()

    def update_env_steps(self, env_steps):
        try:
            self.task_queue.put_nowait((TaskType.UPDATE_ENV_STEPS, env_steps))
        except Full:
            pass

    def join(self):
        timeout = 1.0
        try:
            self.process.wait(timeout)
        except subprocess.TimeoutExpired:
            if self.process_is_alive():
                log.warning('Process %r could not join, kill it with fire!', self.process)
                self.process.kill()
                log.warning('Process %r is dead (%r)', self.process, self.process_is_alive())            

    def process_is_alive(self):
        return self.process.poll() is None
    
    def kill(self):
        self.process.kill()