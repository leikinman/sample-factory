from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([

])

_experiment = Experiment(
    'k8s-atari-pong-stop-value',
    'python -m sample_factory.algorithms.appo.train_appo --env=atari_pong --algo=APPO --num_workers=10 --max_num_workers=15 --num_envs_per_worker=20 --experiment=k8s-atari-pong-stop-value --set_workers_cpu_affinity=False --platform=k8s --train_dir=/lustre --stop_value 20',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('k8s-atari-pong-stop-value', experiments=[_experiment])