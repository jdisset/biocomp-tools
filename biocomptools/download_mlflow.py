import mlflow
import re
import fnmatch
import os

import sys
from typing import Annotated, List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field, field_validator
from dracon import Arg, make_program
from mlflow.entities import FileInfo
from mlflow.tracking import MlflowClient
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
_is_main_script = False


def _sanitize_path_component(name: str) -> str:
    return re.sub(r'[^\w\-_\. ]', '_', name)


class ArtifactDownloaderConfig(BaseModel):
    mlflow_tracking_uri: Annotated[
        str, Arg(positional=True, help="mlflow tracking server uri.")
    ] = 'https://mlf.rachael.jdisset.com'
    experiment_name_regex: Annotated[
        str, Arg(short="e", help="regex to filter experiment names.")
    ] = Field(default='.*')
    run_name_regex: Annotated[str, Arg(short="r", help="regex to filter run names.")] = Field(
        default=".*"
    )
    artifact_glob_pattern: Annotated[
        str,
        Arg(
            short="a",
            help="glob pattern for download (e.g., 'model/*.pkl', '**/*.json', 'my_folder/').",
        ),
    ] = Field(default='**/*.bestmodel.pickle')
    output_dir: Annotated[
        Path | str, Arg(short="o", help="output directory for downloaded artifacts.")
    ] = './'
    max_parallel_runs: Annotated[
        int, Arg(short="p", help="maximum number of runs to process in parallel.")
    ] = Field(default=4)

    verbose: Annotated[bool, Arg(short="v", help="Enable verbose logging.")] = Field(default=False)

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        from rich.logging import RichHandler

        logging.basicConfig(
            level=logging.INFO if not self.verbose else logging.DEBUG,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)],
        )

    @field_validator('output_dir', mode='before')
    @classmethod
    def _resolve_output_dir(cls, v: Any) -> Path:
        return Path(v).expanduser().resolve()

    def _list_all_artifacts_recursive(
        self, client: MlflowClient, run_id: str, current_relative_path: Optional[str] = None
    ) -> List[FileInfo]:
        all_items: List[FileInfo] = []
        items_at_level = client.list_artifacts(run_id=run_id, path=current_relative_path)

        for item_info in items_at_level:
            all_items.append(item_info)
            if item_info.is_dir:
                all_items.extend(self._list_all_artifacts_recursive(client, run_id, item_info.path))
        return all_items

    def _determine_paths_to_download(
        self, all_artifacts: List[FileInfo], glob_pattern: str
    ) -> List[str]:
        if not all_artifacts:
            return []

        path_to_fileinfo_map: Dict[str, FileInfo] = {fi.path: fi for fi in all_artifacts}

        matched_artifact_paths = {
            art_info.path
            for art_info in all_artifacts
            if fnmatch.fnmatch(art_info.path, glob_pattern)
        }

        if not matched_artifact_paths:
            return []

        final_paths_to_download = set()
        # sort by length to process shorter (parent) paths first if names are similar then by name
        sorted_paths = sorted(list(matched_artifact_paths), key=lambda p: (len(p), p))

        for path_str in sorted_paths:
            is_already_covered = any(
                path_to_fileinfo_map[existing_dl_path].is_dir
                and path_str.startswith(existing_dl_path.rstrip('/') + '/')
                for existing_dl_path in final_paths_to_download
            )
            if not is_already_covered:
                final_paths_to_download.add(path_str)

        return sorted(list(final_paths_to_download), key=lambda p: (len(p), p))

    def _download_batch_for_run(
        self,
        client: MlflowClient,
        run_id: str,
        artifact_paths_to_download: List[str],
        base_download_dir: Path,
    ) -> int:
        downloaded_count = 0
        for artifact_target_path in artifact_paths_to_download:
            # base_download_dir is like .../exp_name/run_name
            # download_artifacts recreates structure of artifact_target_path under base_download_dir
            logger.debug(f"downloading '{artifact_target_path}' for run {run_id}")
            try:
                client.download_artifacts(
                    run_id=run_id, path=artifact_target_path, dst_path=str(base_download_dir)
                )
                downloaded_location = base_download_dir / artifact_target_path
                logger.debug(f"  downloaded to: {downloaded_location}")
                downloaded_count += 1
            except Exception as e:
                logger.warning(
                    f"couldn't get {run_id}/{artifact_target_path}: {type(e).__name__} {e}"
                )
        return downloaded_count

    def _process_single_run(
        self,
        exp_name: str,
        run_id: str,
        run_identifier_for_path: str,
        client: MlflowClient,
    ) -> Tuple[str, str, int]:  # exp_name, run_id, num_downloaded
        logger.debug(
            f"evaluating run: id={run_id} (name: {run_identifier_for_path}) in exp: {exp_name}"
        )

        all_run_artifacts = self._list_all_artifacts_recursive(client, run_id)
        if not all_run_artifacts:
            logger.debug(f"no artifacts found for run {run_id}")
            return exp_name, run_id, 0

        paths_to_download = self._determine_paths_to_download(
            all_run_artifacts, self.artifact_glob_pattern
        )
        if not paths_to_download:
            logger.debug(
                f"no artifacts matched glob '{self.artifact_glob_pattern}' in run {run_id}"
            )
            return exp_name, run_id, 0

        safe_exp_name = _sanitize_path_component(exp_name)
        current_run_download_dir = self.output_dir / safe_exp_name / run_identifier_for_path
        current_run_download_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(
            f"Target paths for run {run_id}: {paths_to_download} into {current_run_download_dir}"
        )

        num_downloaded = self._download_batch_for_run(
            client, run_id, paths_to_download, current_run_download_dir
        )

        return exp_name, run_id, num_downloaded

    def download_mlflow_artifacts(self):
        if _is_main_script:
            logger.info(f"mlflow tracking uri: {self.mlflow_tracking_uri}")
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        client = MlflowClient()

        client.search_experiments(max_results=1)  # just a check

        exp_regex = re.compile(self.experiment_name_regex)
        run_regex = re.compile(self.run_name_regex)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if _is_main_script:
            logger.info(f"artifacts will be downloaded under: {self.output_dir}")

        run_tasks_args = []

        all_experiments = client.search_experiments()
        if not all_experiments:
            logger.warning("no experiments found on the mlflow server.")
            return

        matched_experiments_names: set[str] = set()

        for exp in all_experiments:
            if not exp_regex.match(exp.name):
                continue

            matched_experiments_names.add(exp.name)
            logger.debug(f"matched experiment: '{exp.name}' (id: {exp.experiment_id})")

            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
            )

            for run in runs:
                logger.debug(f"matched run: '{run}'")
                run_id = run.info.run_id
                run_name_from_tag = run.data.tags.get("mlflow.runName")
                run_identifier_for_regex = run_name_from_tag if run_name_from_tag else run_id
                if not run_regex.match(run_identifier_for_regex):
                    continue
                run_identifier_for_path = _sanitize_path_component(run_identifier_for_regex)
                run_tasks_args.append((exp.name, run_id, run_identifier_for_path, client))

        total_downloaded_artifacts = 0
        processed_runs_count = 0

        if not run_tasks_args:
            if not matched_experiments_names:
                logger.warning(f"no experiments matched regex '{self.experiment_name_regex}'")
            else:
                logger.warning(
                    f"found {len(matched_experiments_names)} xp, but no runs matching '{self.run_name_regex}'"
                )
            return

        with ThreadPoolExecutor(max_workers=self.max_parallel_runs) as executor:
            futures = {
                executor.submit(self._process_single_run, *args): args for args in run_tasks_args
            }
            for future in as_completed(futures):
                _exp_name, _run_id, num_downloaded = (0, 0, 0)
                try:
                    _exp_name, _run_id, num_downloaded = future.result()
                    total_downloaded_artifacts += num_downloaded
                    processed_runs_count += 1
                    logger.debug(
                        f"run {_run_id} from exp {_exp_name} downloaded {num_downloaded} items."
                    )

                except Exception as e:
                    args_for_failed_task = futures[
                        future
                    ]  # (xp_name, run_id, run_id_for_path, clnt)
                    logger.error(
                        f"error processing run {args_for_failed_task[1]} from exp {args_for_failed_task[0]}: {type(e).__name__} {e}",
                    )

        print(
            f"Downloaded {total_downloaded_artifacts} artifact(s) from {processed_runs_count} run(s) "
            f"across {len(matched_experiments_names)} experiment(s).",
            file=sys.stdout,
        )
        print(f"All artifacts are saved under: {self.output_dir}", file=sys.stdout)


def main():
    global _is_main_script
    _is_main_script = True

    program = make_program(
        ArtifactDownloaderConfig,
        name="mlf-dl",
        description="downloads artifacts from mlflow runs.",
    )
    cli_config, _ = program.parse_args(sys.argv[1:])
    cli_config.download_mlflow_artifacts()


if __name__ == "__main__":
    main()
