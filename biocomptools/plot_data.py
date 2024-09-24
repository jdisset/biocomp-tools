# {{{                        --     imports     -
import ray
import biocomp.utils as ut
import biocomptools.toollib.old_plot as pl
from pathlib import Path
import hydra
import rich
import logging
import argparse
from omegaconf import OmegaConf
import dracon as dr

##────────────────────────────────────────────────────────────────────────────}}}


def setup_logging(loglevel=logging.WARNING):
    import warnings

    warnings.filterwarnings("ignore", message=".*Defaults list is missing")
    warnings.filterwarnings("ignore", message=".*fork() was called")
    import logging

    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=loglevel,
    )
    for name in [
        'biocomp',
        'matplotlib',
        'PIL',
        'biocomptools',
        'hydra',
        'omegaconf',
        'jax',
        'ray',
    ]:
        ut.set_loglevel(name, loglevel)


setup_logging()


def get_job_tasks(job_file: Path) -> list[pl.FigureTask]:
    conf = dr.load(job_file, raw_dict=True)
    oconf = OmegaConf.create(conf)
    job = pl.PlotJob.model_validate(oconf)
    return job.generate_figure_tasks()


def run_task(task: pl.FigureTask):
    task.run()


@ray.remote
class TaskActor:
    def __init__(self, task: pl.FigureTask):
        self.task = task

    def run(self):
        setup_logging()
        logging.info(f'Running task with {len(self.task.data)} data sources')
        self.task.run()


def main():
    parser = argparse.ArgumentParser(description='Make plots')
    parser.add_argument(
        '--job_file',
        type=str,
        help='path to yaml job file',
    )
    parser.add_argument(
        '--job_list',
        help='path to a txt file with multiple job files',
    )
    parser.add_argument(
        '--num_cpus',
        type=int,
        default=8,
        help='number of cpus to use',
    )

    args = parser.parse_args()

    setup_logging()
    tasks = []

    if args.job_file:
        tasks = get_job_tasks(Path(args.job_file))

    if args.job_list:
        with open(args.job_list, 'r') as f:
            job_files = f.read().splitlines()
        for job_file in job_files:
            tasks += get_job_tasks(Path(job_file))

    tasks = ut.flatten(tasks)

    print(f'Found {len(tasks)} tasks')

    if not tasks:
        raise ValueError('No tasks found')

    if args.num_cpus > 1:
        ray.init(num_cpus=args.num_cpus)
        t_actors = [TaskActor.remote(task) for task in tasks]
        futures = [ta.run.remote() for ta in t_actors]
        ray.get(futures)
    else:
        for task in tasks:
            task.run()


if __name__ == '__main__':
    main()
