import fire

from core.trainer import Trainer
from my_utils.parse_experiment import parse_experiment


@parse_experiment
def train(
    trainer: Trainer,
    **experiment,
):

    trainer(**experiment)


if __name__ == "__main__":
    fire.Fire(train)
