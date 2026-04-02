import asyncio
from asyncio import run
from logging import DEBUG, INFO, WARNING, basicConfig, getLogger

import click

logger = getLogger(__name__)
_CLI_VERBOSITY = 0


def _configure_logging(verbosity: int) -> None:
    level = DEBUG if verbosity >= 2 else INFO if verbosity == 1 else WARNING
    basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    # Keep babelbit logs visible even if third-party libs mutate logger levels.
    getLogger("babelbit").setLevel(level)
    getLogger("babelbit.cli.runner").setLevel(level)
    getLogger("babelbit.utils.predict_utterances").setLevel(level)
    getLogger("babelbit.utils.predict_engine").setLevel(level)
    getLogger("babelbit.utils.managed_container_registry").setLevel(level)


@click.group(name="bb")
@click.option(
    "-v",
    "--verbosity",
    count=True,
    help="Increase verbosity (-v INFO, -vv DEBUG)",
)
def cli(verbosity: int):
    """Babelbit CLI"""
    from babelbit.utils.settings import get_settings
    global _CLI_VERBOSITY
    _CLI_VERBOSITY = verbosity

    settings = get_settings()
    _configure_logging(_CLI_VERBOSITY)
    logger.debug("Babelbit started (version=%s)", settings.BABELBIT_VERSION)


@cli.command("runner")
def runner_cmd():
    """Launches runner every TEMPO blocks."""
    from babelbit.cli.runner import runner_loop

    _configure_logging(_CLI_VERBOSITY)
    logger.info("Starting runner command")
    asyncio.run(runner_loop())


@cli.command("signer")
def signer_cmd():
    from babelbit.cli.signer_api import run_signer

    _configure_logging(_CLI_VERBOSITY)
    logger.info("Starting signer command")
    asyncio.run(run_signer())


@cli.command("subtensor-gateway")
def subtensor_gateway_cmd():
    from babelbit.cli.subtensor_gateway import run_subtensor_gateway

    _configure_logging(_CLI_VERBOSITY)
    logger.info("Starting subtensor gateway command")
    asyncio.run(run_subtensor_gateway())


@cli.command("validate")
@click.option(
    "--tail", type=int, envvar="BABELBIT_TAIL", default=28800, show_default=True
)
@click.option(
    "--alpha", type=float, envvar="BABELBIT_ALPHA", default=0.2, show_default=True
)
@click.option(
    "--m-min", type=int, envvar="BABELBIT_M_MIN", default=25, show_default=True
)
@click.option(
    "--tempo", type=int, envvar="BABELBIT_TEMPO", default=100, show_default=True
)
def validate_cmd(tail: int, alpha: float, m_min: int, tempo: int):
    """
    Babelbit validator (mainnet cadence):
      - attend block%tempo==0
      - calcule (uids, weights) winner-takes-all
      - push via signer, fallback local si signer HS
    """
    from babelbit.cli.validate import _validate_main
    from babelbit.utils.prometheus import _start_metrics

    _configure_logging(_CLI_VERBOSITY)
    logger.info("Starting validate command")
    _start_metrics()
    # Overriding to keep validators in sync
    tempo = 100
    asyncio.run(_validate_main(tail=tail, alpha=alpha, m_min=m_min, tempo=tempo))


# TODO: remove this later
@cli.command("test-metagraph")
def test_metagraph_cmd():
    from babelbit.utils.bittensor_helpers import test_metagraph

    _configure_logging(_CLI_VERBOSITY)
    run(test_metagraph())


if __name__ == "__main__":
    _configure_logging(1)

    cli()
