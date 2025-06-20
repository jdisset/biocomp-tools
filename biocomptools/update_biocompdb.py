import logging
import shutil
import sqlite3
import xxhash
import sys
import traceback
import json5
import os
import pickle
import json
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import fnmatch
from typing import Annotated, Any, Dict, List, Optional, Tuple, TypeAlias

from rich.logging import RichHandler
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sqlmodel import Session
import pandas as pd
from PIL import Image
import pikepdf
from pydantic import Field, BaseModel, PrivateAttr, model_validator

import biocomp.utils as ut
import biocomptools.toollib.models as md
from biocomptools.toollib.common import config
from dracon.commandline import make_program, Arg

ProcessDatafileResult: TypeAlias = Tuple[md.DataFile, str, Dict[str, Any]]

ModelLoadResult: TypeAlias = Tuple[md.TrainedModel, List[md.Metric]]
FigureProcessorResult: TypeAlias = Tuple[Optional[md.Figure], List[md.Plot], List[md.Metric]]
ExperimentWorkerResult: TypeAlias = Tuple[
    str, Optional[md.Experiment], List[Dict[str, Any]], List[Tuple[str, str]]
]

RICH_PROGRESS_COLUMNS = [
    SpinnerColumn(style="progress.spinner"),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(bar_width=None),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TextColumn("({task.completed} of {task.total})"),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
]


class WorkerError(Exception):
    def __init__(self, message: str, path_info: Optional[str | Path] = None):
        super().__init__(f"{message}{f' (path: {path_info})' if path_info else ''}")
        self.path_info = str(path_info) if path_info else None


class ModelLoadError(WorkerError):
    pass


class DatafileProcessingError(WorkerError):
    pass


class FigureProcessingError(WorkerError):
    pass


class ExperimentWorkerArgs(BaseModel):
    exp_model: md.Experiment
    base_dir_s: str
    recipe_rel_subpath_s: str
    config_calib_paths: List[str]
    metadata_exclusion_patterns: List[str]
    n_inner_workers: int
    delete_existing: bool = True
    verbose: bool
    model_config = {'arbitrary_types_allowed': True}


def _w_load_model(args: Tuple[str, str]) -> ModelLoadResult:
    mpath_s, base_dir_s = args
    mpath, base_dir = Path(mpath_s), Path(base_dir_s)
    try:
        with open(mpath, 'rb') as f:
            biocomp_model = pickle.load(f)
        metadata = getattr(biocomp_model, 'metadata', {})
        training_set = [
            md.NetworkDataPair(**e)
            for e in metadata.get('training_set', [])
            if isinstance(e, dict) and 'network_name' in e and 'datafile_path' in e
        ]
        
        model_name = getattr(biocomp_model, 'signature', mpath.stem)
        trained_model = md.TrainedModel(
            name=model_name,
            path_to_model=mpath.relative_to(base_dir).as_posix(),
            run_name=metadata.get('run_name', None),
            experiment_name=metadata.get('experiment_name', None),
            end_loss=metadata.get('end_loss', None),
            training_config=metadata,
            training_set=training_set,
        )
        
        # Extract metrics from metadata
        metrics = []
        
        # Add end_loss as a training loss metric if it exists
        if metadata.get('end_loss') is not None:
            metrics.append(md.Metric(
                trained_model_name=model_name,
                metric_type="training_loss",
                metric_name="final",
                numeric_value=float(metadata['end_loss']),
                timestamp=metadata.get('end_time'),
                meta={"source": "training_completion"}
            ))
        
        # Extract logger metrics
        logger_metrics = metadata.get('logger_metrics', [])
        for logger_data in logger_metrics:
            if not isinstance(logger_data, dict):
                continue
                
            for logger_name, logger_values in logger_data.items():
                if not isinstance(logger_values, dict):
                    continue
                    
                # Handle validation loss loggers
                if "validation_loss" in logger_name:
                    # Extract validation set info from loggers metadata
                    validation_set_name = None
                    loggers_meta = metadata.get('loggers', [])
                    for logger_meta in loggers_meta:
                        if (isinstance(logger_meta, dict) and 
                            logger_meta.get('validation_name') == logger_name.split('_validation_loss')[0]):
                            validation_set_name = logger_meta.get('validation_set', {}).get('name')
                            break
                    
                    # Add RMSE metric
                    if 'RMSE' in logger_values:
                        metrics.append(md.Metric(
                            trained_model_name=model_name,
                            metric_type="validation_loss",
                            metric_name=logger_name,
                            numeric_value=float(logger_values['RMSE']),
                            meta={
                                "validation_set": validation_set_name,
                                "logger_name": logger_name.split('_validation_loss')[0]
                            }
                        ))
                    
                    # Add per-network metrics
                    if 'per_network' in logger_values:
                        for network_name, network_metrics in logger_values['per_network'].items():
                            if 'RMSE' in network_metrics:
                                metrics.append(md.Metric(
                                    trained_model_name=model_name,
                                    metric_type="validation_loss_per_network",
                                    metric_name=f"{logger_name}_{network_name}",
                                    numeric_value=float(network_metrics['RMSE']),
                                    meta={
                                        "validation_set": validation_set_name,
                                        "logger_name": logger_name.split('_validation_loss')[0],
                                        "network_name": network_name
                                    }
                                ))
                
                # Handle other logger types (training loss, gradient norms, etc.)
                else:
                    # For other numeric metrics
                    if isinstance(logger_values, (int, float)):
                        metrics.append(md.Metric(
                            trained_model_name=model_name,
                            metric_type="logger_metric",
                            metric_name=logger_name,
                            numeric_value=float(logger_values),
                            meta={"logger_name": logger_name}
                        ))
                    elif isinstance(logger_values, dict):
                        # Store complex logger data as JSON
                        metrics.append(md.Metric(
                            trained_model_name=model_name,
                            metric_type="logger_metric",
                            metric_name=logger_name,
                            json_value=logger_values,
                            meta={"logger_name": logger_name}
                        ))
        
        return trained_model, metrics
        
    except Exception as e:
        raise ModelLoadError(f"failed to load model: {e}", path_info=mpath) from e



def _w_parse_xp_file(args: Tuple[str, str]) -> Tuple[str, Optional[md.Experiment]]:
    xp_dir_s, base_dir_s = args
    xp_dir, base_dir = Path(xp_dir_s), Path(base_dir_s)
    xp_meta_file = xp_dir / 'experiment.json5'

    get_rel_path = lambda p, base: (
        p.relative_to(base).as_posix() if p.is_relative_to(base) else p.as_posix()
    )

    if not xp_meta_file.exists():
        return xp_dir.name, None
    try:
        content = json5.loads(xp_meta_file.read_text())
        return xp_dir.name, md.Experiment(
            name=xp_dir.name, path=get_rel_path(xp_dir, base_dir), content=content, errors={}
        )
    except Exception as e:
        return xp_dir.name, md.Experiment(
            name=xp_dir.name,
            path=get_rel_path(xp_dir, base_dir),
            content={},
            errors={
                'parsing_json5': [
                    f"File: {xp_meta_file.name}, Error: {e}",
                    traceback.format_exc(limit=1),
                ]
            },
        )


def _validate_df_attrs(attrs: Any, dfile_name: str, xp_name_ctx: str) -> List[str]:
    if not isinstance(attrs, dict):
        return [f"datafile {dfile_name} attrs not dict (is {type(attrs)})"]
    errs = []
    spec = {'calibration': ('namehash', dict), 'sample': ('recipe', dict), 'xp': ('name', dict)}
    for key, (subkey, val_type) in spec.items():
        attr_val = attrs.get(key)
        if not (isinstance(attr_val, val_type) and subkey in attr_val):
            errs.append(
                f"datafile {dfile_name} invalid/missing '{key}' (must be {val_type.__name__} with '{subkey}')"
            )
    if not errs and attrs.get('xp', {}).get('name') != xp_name_ctx:
        errs.append(
            f"exp name mismatch in {dfile_name}: file '{attrs.get('xp', {}).get('name')}', context '{xp_name_ctx}'"
        )
    return errs


def _core_proc_df(df_path: Path, base_dir: Path, xp_name: str) -> ProcessDatafileResult:
    try:
        df_content = pd.read_parquet(df_path)
        if err_msgs := _validate_df_attrs(df_content.attrs, df_path.name, xp_name):
            raise DatafileProcessingError(
                f"{df_path.name}: {'; '.join(err_msgs)}", path_info=df_path
            )

        cal_attrs = df_content.attrs['calibration']
        namehash = cal_attrs['namehash']
        cal_full_name = f"{xp_name}_{namehash}"
        dfile_obj = md.DataFile(
            file=df_path.relative_to(base_dir).as_posix(),
            attrs=df_content.attrs,
            calibration_name=cal_full_name,
            priority=1000 if (df_path.parent / '.mark_favorite').exists() else 0,
        )
        return dfile_obj, cal_full_name, cal_attrs.get('pipeline', {})
    except Exception as e:
        if isinstance(e, DatafileProcessingError):
            raise
        raise DatafileProcessingError(f"processing error: {e}", path_info=df_path) from e


def _extract_file_meta(file_path: Path) -> dict:
    meta = {}
    fmt = lambda v: (
        [str(i) for i in v]
        if isinstance(v, pikepdf.Array)
        else str(v)
        if isinstance(v, pikepdf.String)
        else v
    )
    try:
        if file_path.suffix.lower() == '.pdf':
            with pikepdf.open(file_path) as pdf:
                meta = {k[1:]: fmt(v) for k, v in pdf.docinfo.items()}
        elif file_path.suffix.lower() == '.png':
            with Image.open(file_path) as img:
                if subj_src := getattr(img, 'text', {}):
                    meta = {'Subject': subj_src.get('Subject', '{}')}
        subject_str = meta.get('Subject', '{}')
        return json.loads(subject_str) if subject_str and subject_str.strip() else {}
    except Exception as e:
        raise FigureProcessingError(f"metadata extraction failed: {e}", path_info=file_path) from e


def _match_path_pat(pattern: str, path_s: str) -> bool:
    p_transformed = (
        pattern.replace('**', '<MULTI>')
        .replace('*', '<SINGLE>')
        .replace('<MULTI>', '*')
        .replace('<SINGLE>', '[^/]*')
    )
    if fnmatch.fnmatch(path_s, p_transformed):
        return True
    if p_transformed.startswith('*/') and path_s == p_transformed[2:]:
        return True
    return bool(
        pattern.startswith('**')
        and p_transformed.startswith('*/')
        and '/' not in path_s
        and fnmatch.fnmatch(path_s, p_transformed[2:])
    )


def _should_exclude_path(path_s: str, patterns: List[str]) -> bool:
    return any(_match_path_pat(p, path_s) for p in patterns)


def clean_dict(data: Any, patterns: List[str], cur_path: str = '') -> Any:
    if isinstance(data, dict):
        cl_d = {
            k: v_cl
            for k, v in data.items()
            if not _should_exclude_path(p := f"{cur_path}/{k}" if cur_path else k, patterns)
            and (v_cl := clean_dict(v, patterns, p)) is not None
        }
        return cl_d or None
    if isinstance(data, list):
        cl_l = [
            i_cl
            for i, item in enumerate(data)
            if not _should_exclude_path(p := f"{cur_path}/{i}", patterns)
            and (i_cl := clean_dict(item, patterns, p)) is not None
        ]
        return cl_l or None
    return data


def _proc_plot_task(
    task: dict, fig_file: str, task_idx: int, patterns: List[str]
) -> Tuple[Optional[md.Plot], List[md.Metric]]:
    ds_type = task.get('datasource_type')
    net_name = task.get('network_name') or task.get('network', {}).get('name')
    metrics = []

    # base plot args
    plot_args = {
        'in_figure': fig_file,
        'position': task_idx,
        'network_name': net_name,
        'plot_method': task.get('plot_method'),
        'input_names': task.get('input_names', []),
        'output_name': task.get('output_name'),
        'datasource_type': ds_type,
        'meta': clean_dict(dict(task), patterns),
    }

    if ds_type == 'database' and (df_path := task.get('datafile', {}).get('file')):
        return md.Plot(from_datafile=df_path, **plot_args), metrics

    elif ds_type == 'prediction' and net_name and (model_sig := task.get('model_signature')):
        pred_stats = task.get('prediction_stats', {})
        
        # Extract metrics from prediction stats and create Metric objects
        plot_source = f"{fig_file}:{task_idx}"
        
        if 'mse' in pred_stats:
            metrics.append(md.Metric(
                trained_model_name=model_sig,
                metric_type="prediction_mse",
                metric_name=f"plot_{net_name}",
                numeric_value=float(pred_stats['mse']),
                plot_source=plot_source,
                meta={
                    "network_name": net_name,
                    "datafile_path": task.get('extra_prediction_info', {}).get('datafile', {}).get('file'),
                    "plot_method": task.get('plot_method')
                }
            ))
        
        if 'grid_mse' in pred_stats:
            metrics.append(md.Metric(
                trained_model_name=model_sig,
                metric_type="prediction_grid_mse", 
                metric_name=f"plot_{net_name}",
                numeric_value=float(pred_stats['grid_mse']),
                plot_source=plot_source,
                meta={
                    "network_name": net_name,
                    "datafile_path": task.get('extra_prediction_info', {}).get('datafile', {}).get('file'),
                    "plot_method": task.get('plot_method')
                }
            ))
        
        normalized_grid_mse = pred_stats.get('normalized_grid_mse', pred_stats.get('grid_mse'))
        if normalized_grid_mse is not None:
            metrics.append(md.Metric(
                trained_model_name=model_sig,
                metric_type="prediction_normalized_grid_mse",
                metric_name=f"plot_{net_name}",
                numeric_value=float(normalized_grid_mse),
                plot_source=plot_source,
                meta={
                    "network_name": net_name,
                    "datafile_path": task.get('extra_prediction_info', {}).get('datafile', {}).get('file'),
                    "plot_method": task.get('plot_method')
                }
            ))
            
        if 'eval_npoints' in pred_stats:
            metrics.append(md.Metric(
                trained_model_name=model_sig,
                metric_type="prediction_n_points",
                metric_name=f"plot_{net_name}",
                numeric_value=float(pred_stats['eval_npoints']),
                plot_source=plot_source,
                meta={
                    "network_name": net_name,
                    "datafile_path": task.get('extra_prediction_info', {}).get('datafile', {}).get('file'),
                    "plot_method": task.get('plot_method')
                }
            ))
        
        # Store any extra stats as JSON metric
        if pred_stats:
            metrics.append(md.Metric(
                trained_model_name=model_sig,
                metric_type="prediction_extra_stats",
                metric_name=f"plot_{net_name}",
                json_value=pred_stats,
                plot_source=plot_source,
                meta={
                    "network_name": net_name,
                    "datafile_path": task.get('extra_prediction_info', {}).get('datafile', {}).get('file'),
                    "plot_method": task.get('plot_method')
                }
            ))

        # Create simplified plot without prediction fields
        return md.Plot(**plot_args), metrics

    return None, metrics



def _w_proc_fig_path(args_tuple: Tuple[str, str, List[str]]) -> FigureProcessorResult:
    fig_path_s, base_dir_s, meta_exclude_patterns = args_tuple
    fig_path, base_dir = Path(fig_path_s), Path(base_dir_s)
    try:
        metadata = _extract_file_meta(fig_path)
        if not metadata:
            return None, [], []

        rel_path = fig_path.relative_to(base_dir).as_posix()
        fig_meta_content = metadata.get('FigureMetadata', {})
        cleaned_fig_meta = {k: v for k, v in metadata.items() if k != 'FigureMetadata'}
        if 'FigureMetadata' in metadata:
            cleaned_fig_meta['FigureMetadata'] = {
                k: v for k, v in fig_meta_content.items() if k != 'plot_tasks'
            }

        figure = md.Figure(file=rel_path, meta=cleaned_fig_meta)
        plots = []
        all_metrics = []
        ptasks = fig_meta_content.get('plot_tasks', [])
        should_debug = False

        if len(ptasks) == 0:
            logging.getLogger('biocomp_db').warning(
                f"No plot tasks found in figure {fig_path.name}. "
                "This may indicate an incomplete or improperly formatted figure."
            )
        elif len(ptasks) > 10:
            logging.getLogger('biocomp_db').warning(
                f"Figure {fig_path.name} has {len(ptasks)} plot tasks."
            )
            should_debug = True

        for i, task_data in enumerate(ptasks):
            plot, task_metrics = _proc_plot_task(task_data, rel_path, i, meta_exclude_patterns)
            if plot:
                plots.append(plot)
                all_metrics.extend(task_metrics)
                if should_debug:
                    print(f"Processed plot {i + 1}/{len(ptasks)}: {plot}")
                    print(f"Plot extracted {len(task_metrics)} metrics")

        if should_debug:
            print(f"Processed {len(plots)} plots and {len(all_metrics)} metrics from {fig_path.name}.")

        return figure, plots, all_metrics
    except Exception as e:
        if isinstance(e, FigureProcessingError):
            raise
        raise FigureProcessingError(f"generic error: {e}", path_info=fig_path) from e



def _w_proc_xp(worker_args: ExperimentWorkerArgs) -> ExperimentWorkerResult:
    xp, args = worker_args.exp_model, worker_args
    base_dir, local_errors = Path(args.base_dir_s), []
    logger = logging.getLogger('biocomp_db')
    try:
        xp.recipes = xp.find_recipes(path_prefix=base_dir, recipe_subpath=args.recipe_rel_subpath_s)
        logger.debug(f"Recipes found by find_recipes for {xp.name}: {len(xp.recipes)}")
    except Exception as e:
        logger.error(f"Error finding recipes for experiment {xp.name}")
        logger.exception(e)
        local_errors.append(('recipe_finding', f"Exp '{xp.name}': {e}"))
        xp.recipes = []

    xp_root_fs = base_dir / xp.path if not Path(xp.path).is_absolute() else Path(xp.path)
    df_paths = sorted(
        {
            p
            for pat in args.config_calib_paths
            for p in (xp_root_fs / pat).rglob('*.parquet')
            if (xp_root_fs / pat).is_dir()
        }
    )

    df_proc_results: List[ProcessDatafileResult] = []
    if df_paths:
        with ThreadPoolExecutor(
            max_workers=args.n_inner_workers, thread_name_prefix=f"DF_{xp.name[:5]}"
        ) as tpe:
            futures = {
                tpe.submit(_core_proc_df, df_p, base_dir, xp.name): df_p for df_p in df_paths
            }
            for fut in as_completed(futures):
                try:
                    df_proc_results.append(fut.result())
                except Exception as e:
                    local_errors.append(
                        ('datafile_item_exception', f"Error processing {futures[fut]}: {e}")
                    )

    cal_map: Dict[str, md.Calibration] = {}
    for dfile, cal_full_name, pipe_info in df_proc_results:
        if cal_full_name not in cal_map:
            cal_map[cal_full_name] = md.Calibration(
                fullname=cal_full_name,
                name=cal_full_name.replace(f"{xp.name}_", "", 1),
                pipeline=pipe_info,
                data_files=[],
            )
        cal_map[cal_full_name].data_files.append(dfile)

    recipe_lookup = {
        r.content['name']: r
        for r in xp.recipes
        if isinstance(r, md.Recipe) and r.content and 'name' in r.content
    }
    for cal in cal_map.values():
        for dfile in cal.data_files:
            if not isinstance(dfile.attrs, dict):
                continue
            try:
                recipe_name = dfile.attrs.get('sample', {}).get('recipe')
                if not recipe_name:
                    local_errors.append(
                        (
                            'recipe_link_missing_in_dfile',
                            f"Missing recipe in {dfile.file} (xp {xp.name})",
                        )
                    )
                    continue
                if not (target_recipe := recipe_lookup.get(recipe_name)):
                    local_errors.append(
                        (
                            'recipe_link_target_not_found',
                            f"Recipe '{recipe_name}' from {dfile.file} not found in {xp.name}.",
                        )
                    )
                    continue
                dfile.recipe_name = target_recipe.name
                if dfile not in target_recipe.data_files:
                    target_recipe.data_files.append(dfile)
            except Exception as e:
                local_errors.append(
                    (
                        'recipe_link_exception',
                        f"Error linking recipe for {dfile.file} (xp {xp.name}): {e}",
                    )
                )
    return xp.name, xp, [c.model_dump() for c in cal_map.values()], local_errors


class BiocompDBUpdater(BaseModel):
    base_dir: Annotated[Path | str, Arg(help="Path to base directory.")] = config.paths.root
    xp_dir: Annotated[Path | str, Arg(help="Path to experiments directory.")] = 'Experiments'
    recipe_relative_subpath: Annotated[str, Arg(help="Relative path to recipe directory.")] = (
        'recipes'
    )
    models_dir: Annotated[str, Arg(help="Models directory name relative to base_dir.")] = "Models"
    plots_dir: Annotated[str, Arg(help="Directory to scan for plots, relative to base_dir.")] = (
        "Plots/Figures"
    )
    verbose: Annotated[bool, Arg(short="v", help="Enable verbose logging.")] = True
    process_experiments: Annotated[bool, Arg(help="Process experiments and recipes")] = True
    process_models: Annotated[bool, Arg(help="Process trained models")] = True
    process_figures: Annotated[bool, Arg(help="Process figures and plots")] = True
    nworkers: Annotated[int, Arg(help="Number of parallel workers")] = 8

    _logger: logging.Logger = PrivateAttr()
    _metadata_exclusion_patterns: List[str] = PrivateAttr(
        default=[
            '**/network',
            '**/built_network',
            '**/network_info',
            '**/file_stem',
            '**/datafile/attrs',
            '**/calibration',
        ]
    )

    @staticmethod
    def _resolve_path(p: Path | str, base: Optional[Path] = None) -> Path:
        path = Path(p)
        return (base / path if base and not path.is_absolute() else path).expanduser().resolve()

    def _configure_logger(self):
        self._logger = logging.getLogger('biocomp_db')
        if self._logger.handlers:
            return
        self._logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        fmt_str = (
            '%(message)s'
            if not self.verbose
            else '%(asctime)s %(levelname)-8s %(name)s: %(message)s [%(threadName)s]'
        )
        rh = RichHandler(
            rich_tracebacks=True,
            tracebacks_show_locals=self.verbose,
            show_time=self.verbose,
            show_path=self.verbose,
            level=logging.DEBUG if self.verbose else logging.INFO,
        )
        rh.setFormatter(logging.Formatter(fmt_str))
        self._logger.addHandler(rh)
        self._logger.propagate = False

        log_dir = self._resolve_path(
            getattr(config.paths, "logs", self.base_dir / 'logs'), self.base_dir
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f'biocomp_db_{datetime.now():%Y%m%d}.log')
        fh.setFormatter(
            logging.Formatter(
                '%(asctime)s-%(name)s-%(levelname)s-%(message)s [%(processName)s-%(threadName)s]'
            )
        )
        fh.setLevel(logging.DEBUG)
        self._logger.addHandler(fh)

    @model_validator(mode='after')
    def _setup(self) -> 'BiocompDBUpdater':
        self.base_dir = self._resolve_path(self.base_dir)
        for fld in ['xp_dir', 'models_dir', 'plots_dir']:
            setattr(self, fld, self._resolve_path(getattr(self, fld), self.base_dir))
        self._configure_logger()
        for lib_name in ['biocomp', 'jax', 'httpx', 'numba', 'PIL.PngImagePlugin']:
            logging.getLogger(lib_name).setLevel(logging.WARNING)
        if self.nworkers <= 0:
            cpus = os.cpu_count() or 1
            self.nworkers = max(
                1, min((cpus - 1) if cpus > 1 else 1, getattr(config.system, "max_workers_cap", 8))
            )
        return self

    def _add_error_to_item(self, item: Any, key: str, detail: str):
        if hasattr(item, 'errors') and isinstance(item.errors, dict):
            item.errors.setdefault(key, []).append(detail)

    def _load_xps(self, progress: Progress) -> Dict[str, md.Experiment]:
        self._logger.info("loading experiments...")
        if not self.xp_dir.is_dir():
            self._logger.error(f"experiments root {self.xp_dir} not found.")
            return {}

        xp_cand_dirs = sorted([d for d in self.xp_dir.iterdir() if d.is_dir()])
        xps: Dict[str, md.Experiment] = {}
        tid = progress.add_task("[cyan]Parsing experiment files...", total=len(xp_cand_dirs))
        with ProcessPoolExecutor(max_workers=self.nworkers) as exe:
            futures = {
                exe.submit(_w_parse_xp_file, (str(d), str(self.base_dir))): d for d in xp_cand_dirs
            }
            for fut in as_completed(futures):
                xp_dir_p = futures[fut]
                try:
                    xp_name, xp_obj = fut.result()
                    if xp_obj is None:
                        self._logger.warning(f'no experiment.json5 in {xp_dir_p.name}, skipping')
                    elif xp_obj:
                        if xp_obj.errors:
                            [
                                self._logger.error(f"Error parsing {xp_name} ({et}): {d}")
                                for et, eds in xp_obj.errors.items()
                                for d in eds
                            ]
                        if xp_name in xps:
                            self._logger.warning(
                                f"Duplicate experiment name {xp_name} from {xp_dir_p}, overwriting."
                            )
                        xps[xp_name] = xp_obj
                except Exception as e:
                    self._logger.error(
                        f"Critical error loading experiment from dir {xp_dir_p}", exc_info=e
                    )
                progress.update(tid, advance=1)
        progress.update(
            tid, description="[cyan]Experiment files parsed.", completed=len(xp_cand_dirs)
        )
        self._logger.info(f"loaded initial data for {len(xps)} experiments")
        return xps

    def _get_db_hash(self, db_fpath: Path) -> str:
        with sqlite3.connect(f"file:{db_fpath}?mode=ro", uri=True) as conn:
            return xxhash.xxh128('\n'.join(conn.iterdump()).encode()).hexdigest()

    def _backup_db(self) -> bool:
        self._logger.info("checking database for backup")
        db_f = self._resolve_path(config.db.sqlite.path, self.base_dir)
        bak_dir = self._resolve_path(config.db.sqlite.backup.dir, self.base_dir)
        if not db_f.exists():
            self._logger.error(f"db file {db_f} not found.")
            return False
        bak_dir.mkdir(parents=True, exist_ok=True)
        if (limit := config.db.sqlite.backup.keep_n) > 0:
            cur_baks = sorted(bak_dir.glob(f'{db_f.stem}_*.sqlite'), key=os.path.getmtime)
            if (num_del := len(cur_baks) - limit + 1) > 0:
                for old in cur_baks[:num_del]:
                    old.unlink(missing_ok=True)
                    self._logger.info(f"rm old backup: {old.name}")
            if cur_baks and cur_baks[-1].exists():  # ensure last backup file exists before hashing
                try:
                    if self._get_db_hash(cur_baks[-1]) == self._get_db_hash(db_f):
                        self._logger.info('no db changes, skipping backup.')
                        return True
                except Exception as e:
                    self._logger.warning(
                        f"could not compare db hashes ({e}), proceeding with backup."
                    )
        new_bkp_f = bak_dir / f'{db_f.stem}_{datetime.now():%Y-%m-%d_%H%M%S}.sqlite'
        try:
            shutil.copy2(db_f, new_bkp_f)
            self._logger.info(f'db backed up to {new_bkp_f}')
            return True
        except Exception as e:
            self._logger.error(f"error copying db for backup: {e}")
            new_bkp_f.unlink(missing_ok=True)
            return False

    def _load_models(self, progress: Progress) -> Tuple[List[md.TrainedModel], List[md.Metric]]:
            self._logger.info("loading models...")
            if not self.models_dir.is_dir():
                self._logger.info(f"Models dir {self.models_dir} not found")
                return [], []
            mpaths = [
                p for p in self.models_dir.rglob('*model.p*kl*') if 'checkpoints/' not in p.as_posix()
            ]
            self._logger.info(f"found {len(mpaths)} model files in {self.models_dir}")
            if not mpaths:
                return [], []
    
            models: List[md.TrainedModel] = []
            all_metrics: List[md.Metric] = []
            tid = progress.add_task("[magenta]Loading models...", total=len(mpaths))
            with ProcessPoolExecutor(max_workers=self.nworkers) as exe:
                futures = {exe.submit(_w_load_model, (str(p), str(self.base_dir))): p for p in mpaths}
                for fut in as_completed(futures):
                    try:
                        model, metrics = fut.result()
                        models.append(model)
                        all_metrics.extend(metrics)
                    except ModelLoadError as e:
                        self._logger.error(f"Error loading model: {e}")  # e already has path
                    except Exception as e:
                        self._logger.error(
                            f"Critical error getting result for model {futures[fut]}", exc_info=e
                        )
                    progress.update(tid, advance=1)
            progress.update(tid, description="[magenta]Models loaded.", completed=len(mpaths))
            self._logger.info(f"Extracted {len(all_metrics)} metrics from {len(models)} models")
            return models, all_metrics



    def _load_figs(self, progress: Progress) -> Tuple[List[md.Figure], List[md.Plot], List[md.Metric]]:
            self._logger.info("loading figures...")
            figs, plots, all_metrics = [], [], []
            if not self.plots_dir.is_dir():
                self._logger.warning(f"Plots dir {self.plots_dir} not found")
                return figs, plots, all_metrics
    
            fig_paths = [
                p
                for pat in ['**/*.pdf', '**/*.png']
                for p in self.plots_dir.rglob(pat)
                if '__archive' not in p.as_posix()
            ]
            self._logger.info(f"found {len(fig_paths)} figure files in {self.plots_dir}")
            if not fig_paths:
                return figs, plots, all_metrics
    
            tid = progress.add_task("[yellow]Loading figures...", total=len(fig_paths))
            with ProcessPoolExecutor(max_workers=self.nworkers) as exe:
                args_list = [
                    (str(fp), str(self.base_dir), self._metadata_exclusion_patterns) for fp in fig_paths
                ]
                futures = {exe.submit(_w_proc_fig_path, arg_set): arg_set[0] for arg_set in args_list}
                for fut in as_completed(futures):
                    try:
                        fig_obj, plots_list, metrics_list = fut.result()
                        if fig_obj:
                            figs.append(fig_obj)
                            plots.extend(plots_list)
                            all_metrics.extend(metrics_list)
                    except FigureProcessingError as e:
                        self._logger.warning(f"Error processing figure: {e}")
                        self._logger.exception(e)
                    except Exception as e:
                        self._logger.error(
                            f"Critical error processing figure future for {futures[fut]}"
                        )
                        self._logger.exception(e)
                    progress.update(tid, advance=1)
            progress.update(tid, description="[yellow]Figures loaded.", completed=len(fig_paths))
            self._logger.info(f"Extracted {len(all_metrics)} metrics from {len(figs)} figures")
            return figs, plots, all_metrics



    def _build_nets_task_tpe(
        self, recipe: md.Recipe, lib: Any, cache_path: Path
    ) -> Tuple[md.Recipe, Any]:
        try:
            return recipe, recipe.build_networks(
                lib, inverse='all', use_cache=cache_path, add_to_self=False
            )
        except Exception as e:
            self._logger.error(f"Exception in build_networks for {recipe.name}: {e}")
            return recipe, traceback.format_exc(limit=1)

    def _commit_db(self, items: List[Any], progress: Progress) -> None:
        db_p = self._resolve_path(config.db.sqlite.path, self.base_dir)
        if getattr(self, "delete_existing", False) and db_p.exists():
            self._logger.warning(f"Deleting existing database at {db_p}")
            db_p.unlink()
        if not db_p.exists():
            self._logger.warning(f"db file {db_p} not found. creating new.")
            md.create_biocompdb_sqlite(db_p, echo=False)
        if not self._backup_db():
            self._logger.warning("db backup failed/skipped. proceeding cautiously.")

        engine = md.get_biocompdb_sqlite_engine(db_p, echo=False)
        self._logger.info("--- starting database commit ---")
        with Session(engine) as sess:
            try:
                tid_commit = progress.add_task(
                    "[db]Merging items...", total=len(items), visible=self.verbose
                )
                for item in items:
                    try:
                        sess.merge(item)
                    except Exception:
                        self._logger.error(
                            f"Error merging {type(item).__name__} '{getattr(item, 'name', getattr(item, 'fullname', str(item)))}'",
                            exc_info=True,
                        )
                    if self.verbose:
                        progress.advance(tid_commit)
                progress.update(
                    tid_commit,
                    description="[db]Items merged.",
                    completed=len(items),
                    visible=False,
                )
                sess.commit()
                self._logger.info("database transaction committed.")
            except Exception:
                self._logger.critical(
                    "critical error during final db commit. rolled back.", exc_info=True
                )
                sess.rollback()
                raise

    def run(self) -> None:
        start_t = time.time()
        self._logger.info(f"--- starting db update run: {datetime.now():%Y-%m-%d %H:%M:%S} ---")
        self._logger.info(f"using {self.nworkers} parallel workers. base: {self.base_dir}")
        proc_cats = [
            c
            for c, f in [
                ("experiments", self.process_experiments),
                ("models", self.process_models),
                ("figures", self.process_figures),
            ]
            if f
        ]
        self._logger.info(
            f"processing: {', '.join(proc_cats) if proc_cats else 'nothing specified'}"
        )
        if not proc_cats:
            self._logger.warning("nothing to process.")
            return

        lib, xps_map, cals_map, models_list, model_metrics_list, figs_list, plots_list, fig_metrics_list = (
            ut.load_lib(),
            {},
            {},
            [],
            [],
            [],
            [],
            [],
        )

        with Progress(*RICH_PROGRESS_COLUMNS, refresh_per_second=4, transient=False) as progress:
            if self.process_experiments:
                xps_map = self._load_xps(progress)
                if xps_map:
                    self._logger.info(
                        f"Processing recipes/datafiles for {len(xps_map)} experiments..."
                    )
                    tid_xp_proc = progress.add_task(
                        "[blue]Processing experiments (recipes/data)...", total=len(xps_map)
                    )

                    xp_w_args = [
                        ExperimentWorkerArgs(
                            exp_model=xp_obj,
                            base_dir_s=str(self.base_dir),
                            recipe_rel_subpath_s=self.recipe_relative_subpath,
                            config_calib_paths=list(config.calib.paths),
                            metadata_exclusion_patterns=self._metadata_exclusion_patterns,
                            n_inner_workers=max(1, self.nworkers // 2 if self.nworkers > 1 else 1),
                            verbose=self.verbose,
                        )
                        for xp_obj in xps_map.values()
                    ]

                    with ProcessPoolExecutor(max_workers=self.nworkers) as ppe:
                        futures = {
                            ppe.submit(_w_proc_xp, args): args.exp_model.name for args in xp_w_args
                        }
                        for fut in as_completed(futures):
                            xp_name_ctx = futures[fut]
                            try:
                                _, processed_xp_obj, new_cal_ds, w_errs = fut.result()
                                if processed_xp_obj:
                                    xps_map[xp_name_ctx] = processed_xp_obj
                                else:
                                    self._logger.error(
                                        f"Experiment worker for {xp_name_ctx} returned no experiment object."
                                    )
                                    continue  # should not happen

                                for err_k, err_m in w_errs:
                                    self._add_error_to_item(xps_map[xp_name_ctx], err_k, err_m)
                                    self._logger.error(
                                        f"Err in xp worker {xp_name_ctx} [{err_k}]: {err_m}"
                                    )
                                for cal_d in new_cal_ds:
                                    cal_o = md.Calibration.model_validate(cal_d)
                                    if cal_o.fullname in cals_map:
                                        cals_map[cal_o.fullname].data_files.extend(
                                            f
                                            for f in cal_o.data_files
                                            if f not in cals_map[cal_o.fullname].data_files
                                        )
                                    else:
                                        cals_map[cal_o.fullname] = cal_o
                            except Exception as e:
                                self._logger.error(
                                    f"Critical err processing result for xp {xp_name_ctx}",
                                    exc_info=e,
                                )
                                if xp_name_ctx in xps_map:
                                    self._add_error_to_item(
                                        xps_map[xp_name_ctx],
                                        "critical_xp_processing",
                                        traceback.format_exc(),
                                    )
                            progress.update(tid_xp_proc, advance=1)
                    progress.update(
                        tid_xp_proc,
                        description="[blue]Experiments (recipes/data) processed.",
                        completed=len(xps_map),
                    )

                    all_recipes = [r for xp_val in xps_map.values() for r in xp_val.recipes]
                    if not all_recipes:
                        self._logger.warning(
                            "no recipes found in any experiments. stopping experiment processing."
                        )
                        return

                    self._logger.info(f"Building networks for {len(all_recipes)} recipes...")
                    tid_net_bld = progress.add_task(
                        "[green]Building networks...", total=len(all_recipes)
                    )
                    cache_p_nets = Path(config.paths.cache.networks)
                    with ThreadPoolExecutor(
                        max_workers=self.nworkers, thread_name_prefix="NetBuild"
                    ) as tpe:
                        futures = {
                            tpe.submit(self._build_nets_task_tpe, r, lib, cache_p_nets): r
                            for r in all_recipes
                        }
                        for fut in as_completed(futures):
                            recipe_ctx = futures[fut]
                            try:
                                _, nets_or_err = fut.result()
                                if isinstance(nets_or_err, str):
                                    self._add_error_to_item(
                                        recipe_ctx, 'net_build_error_str', nets_or_err
                                    )
                                    self._logger.warning(
                                        f"Failed to build network for {recipe_ctx.name}: {nets_or_err[:200]}"
                                    )
                                elif isinstance(nets_or_err, list):
                                    recipe_ctx.networks.extend(nets_or_err)
                                else:
                                    self._add_error_to_item(
                                        recipe_ctx,
                                        'net_build_payload_unexpected',
                                        f"Type: {type(nets_or_err)}",
                                    )
                            except Exception as e:
                                self._logger.error(
                                    f"Critical err TPE future for {recipe_ctx.name}", exc_info=e
                                )
                                self._add_error_to_item(
                                    recipe_ctx, 'net_build_critical_future', traceback.format_exc()
                                )
                            progress.update(tid_net_bld, advance=1)
                    progress.update(
                        tid_net_bld,
                        description="[green]Networks built.",
                        completed=len(all_recipes),
                    )
                else:
                    self._logger.warning("no experiments loaded to process.")

            if self.process_models:
                models_list, model_metrics_list = self._load_models(progress)
            if self.process_figures:
                figs_list, plots_list, fig_metrics_list = self._load_figs(progress)

            items_commit = [
                i
                for grp in [
                    list(xps_map.values()),
                    list(cals_map.values()),
                    models_list,
                    model_metrics_list,
                    figs_list,
                    plots_list,
                    fig_metrics_list,
                ]
                for i in grp
                if i
            ]
            if items_commit:
                self._commit_db(items_commit, progress)
            else:
                self._logger.info("no items to commit to database.")

        self._logger.info(f"--- db update run finished in {time.time() - start_t:.2f}s ---")


def main() -> None:
    cli_prog = make_program(
        BiocompDBUpdater,
        name='biocomp-dbupdate',
        description='Update Biocomp database from biocomp data folder.',
    )
    updater, _ = cli_prog.parse_args(sys.argv[1:], capture_globals=False)  # type: ignore[misc]
    updater.run()


if __name__ == "__main__":
    main()
