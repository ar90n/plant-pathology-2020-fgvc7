import click

from kedro.framework.cli.utils import (
    KedroCliError,
    _config_file_callback,
    _get_values_as_tuple,
    _reformat_load_versions,
    _split_params,
    env_option,
    split_string,
)
from kedro.framework.session import KedroSession
from kedro.utils import load_obj

from kedro.framework.cli.project import (
    FROM_INPUTS_HELP,
    TO_OUTPUTS_HELP,
    FROM_NODES_HELP,
    TO_NODES_HELP,
    NODE_ARG_HELP,
    RUNNER_ARG_HELP,
    PARALLEL_ARG_HELP,
    ASYNC_ARG_HELP,
    TAG_ARG_HELP,
    LOAD_VERSION_HELP,
    CONFIG_FILE_HELP,
    PIPELINE_ARG_HELP,
    PARAMS_ARG_HELP,
)


def wrap_run_only_missing_runner(runner_cls):
    from typing import Dict, Any
    from kedro.io import DataCatalog
    from kedro.pipeline import Pipeline

    class RuOnlyMissingRunner(runner_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._call_from_run_only_missing = False

        def run(
            self, pipeline: Pipeline, catalog: DataCatalog, run_id: str = None
        ) -> Dict[str, Any]:
            if self._call_from_run_only_missing:
                return super().run(pipeline, catalog, run_id)

            with self:
                return self.run_only_missing(pipeline, catalog)

        def __enter__(self):
            self._call_from_run_only_missing = True

        def __exit__(self, _exc_type, _exc_value, _traceback):
            self._call_from_run_only_missing = False

    return RuOnlyMissingRunner


@click.group(name="Kedro")
def cli(*args, **kwargs):
    pass


@cli.command()
@click.option("--run-only-missing", "run_only_missing", is_flag=True, multiple=False, help="Run only output missing nodes")
@click.option(
    "--from-inputs", type=str, default="", help=FROM_INPUTS_HELP, callback=split_string
)
@click.option(
    "--to-outputs", type=str, default="", help=TO_OUTPUTS_HELP, callback=split_string
)
@click.option(
    "--from-nodes", type=str, default="", help=FROM_NODES_HELP, callback=split_string
)
@click.option(
    "--to-nodes", type=str, default="", help=TO_NODES_HELP, callback=split_string
)
@click.option("--node", "-n", "node_names", type=str, multiple=True, help=NODE_ARG_HELP)
@click.option(
    "--runner", "-r", type=str, default=None, multiple=False, help=RUNNER_ARG_HELP
)
@click.option("--parallel", "-p", is_flag=True, multiple=False, help=PARALLEL_ARG_HELP)
@click.option("--async", "is_async", is_flag=True, multiple=False, help=ASYNC_ARG_HELP)
@env_option
@click.option("--tag", "-t", type=str, multiple=True, help=TAG_ARG_HELP)
@click.option(
    "--load-version",
    "-lv",
    type=str,
    multiple=True,
    help=LOAD_VERSION_HELP,
    callback=_reformat_load_versions,
)
@click.option("--pipeline", type=str, default=None, help=PIPELINE_ARG_HELP)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help=CONFIG_FILE_HELP,
    callback=_config_file_callback,
)
@click.option(
    "--params", type=str, default="", help=PARAMS_ARG_HELP, callback=_split_params
)
# pylint: disable=too-many-arguments,unused-argument,too-many-locals
def run(
    tag,
    env,
    parallel,
    runner,
    run_only_missing,
    is_async,
    node_names,
    to_nodes,
    from_nodes,
    from_inputs,
    to_outputs,
    load_version,
    pipeline,
    config,
    params,
):
    """Run the pipeline."""
    if parallel and runner:
        raise KedroCliError(
            "Both --parallel and --runner options cannot be used together. "
            "Please use either --parallel or --runner."
        )
    runner = runner or "SequentialRunner"
    if parallel:
        deprecation_message = (
            "DeprecationWarning: The behaviour of --parallel and -p flags will change. "
            "In Kedro 0.18.0, `-p` will be an alias for `--pipeline` and the "
            "`--parallel` flag will no longer exist. Instead, the parallel runner "
            "should be used by specifying `--runner=ParallelRunner` (or "
            "`-r ParallelRunner`)."
        )
        click.secho(deprecation_message, fg="red")
        runner = "ParallelRunner"
    runner_class = load_obj(runner, "kedro.runner")

    tag = _get_values_as_tuple(tag) if tag else tag
    node_names = _get_values_as_tuple(node_names) if node_names else node_names

    if run_only_missing:
        runner_class = wrap_run_only_missing_runner(runner_class)
    with KedroSession.create(env=env, extra_params=params) as session:
        session.run(
            tags=tag,
            runner=runner_class(is_async=is_async),
            node_names=node_names,
            from_nodes=from_nodes,
            to_nodes=to_nodes,
            from_inputs=from_inputs,
            to_outputs=to_outputs,
            load_versions=load_version,
            pipeline_name=pipeline,
        )
