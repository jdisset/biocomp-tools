"""biocomp-model-prepare: Prepare a model for distribution/archiving.

Creates a clean output directory with:
- Model file renamed to signature.pickle
- metadata.yaml with model metadata
- Inner nodes and benchmark summary plots
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional

import yaml
from pydantic import BaseModel, ConfigDict

from biocomptools.logging_config import get_logger, setup_logging
from biocomptools.modelmodel import BiocompModel
from dracon.commandline import Arg, make_program

setup_logging(force=False)
logger = get_logger(__name__)


def make_yaml_serializable(obj):
    """Convert objects to YAML-serializable types."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: make_yaml_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_yaml_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, 'model_dump'):
        return make_yaml_serializable(obj.model_dump())
    return obj


class ModelPrepareConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    model_file: Annotated[str, Arg(short='m', help='Path to input model file (.pickle)')]
    output_dir: Annotated[str, Arg(short='o', help='Output directory for prepared model')]
    skip_plots: Annotated[bool, Arg(help='Skip generating plots')] = False
    dataset_file: Annotated[Optional[str], Arg(short='d', help='Dataset file for benchmark')] = None

    def run(self):
        model_path = Path(self.model_file).expanduser().resolve()
        output_base = Path(self.output_dir).expanduser().resolve()

        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            sys.exit(1)

        logger.info(f"Loading model from {model_path}")
        model = BiocompModel.load(model_path)
        sig = model.signature

        logger.info(f"Model signature: {sig}")

        # create output directory
        out_dir = output_base / sig
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {out_dir}")

        # copy model with signature as filename
        model_out = out_dir / f"{sig}.pickle"
        shutil.copy2(model_path, model_out)
        logger.info(f"Copied model to {model_out}")

        # create metadata.yaml
        meta_out = out_dir / "metadata.yaml"
        clean_meta = make_yaml_serializable(model.metadata)
        clean_meta['signature'] = sig
        clean_meta['original_file'] = str(model_path)
        with open(meta_out, 'w') as f:
            yaml.dump(
                clean_meta,
                f,
                default_flow_style=False,
                allow_unicode=True,
                width=120,
                sort_keys=False,
            )
        logger.info(f"Wrote metadata to {meta_out}")

        if self.skip_plots:
            logger.info("Skipping plot generation (--skip-plots)")
            return

        # generate plots
        self._run_plot('innernodes', model_path, out_dir)
        self._run_plot('benchmark_summary', model_path, out_dir)

        # flatten any nested directories created by plots
        self._flatten_output(out_dir, sig)

        logger.info(f"Done! Output in {out_dir}")

    def _flatten_output(self, out_dir: Path, sig: str):
        """Move files from nested signature directories to top level."""
        nested = out_dir / sig
        if nested.exists() and nested.is_dir():
            for f in nested.iterdir():
                if f.is_file():
                    dest = out_dir / f.name
                    shutil.move(str(f), str(dest))
                    logger.debug(f"Moved {f.name} to output directory")
            nested.rmdir()

    def _run_plot(self, plot_name: str, model_path: Path, out_dir: Path):
        logger.info(f"Generating {plot_name} plot...")
        # biocomp-jobs is in the source directory (parent of biocomp-tools)
        import biocomptools

        source_dir = Path(biocomptools.__file__).parent.parent.parent
        cmd = [
            'biocomp-plot',
            f'+biocomp-jobs/plot/{plot_name}.yaml',
            f'++model_path={model_path}',
            f'++output_dir={out_dir}',
        ]
        if plot_name == 'benchmark_summary' and self.dataset_file:
            cmd.append(f'++dataset_file={self.dataset_file}')

        logger.debug(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=source_dir)
        if result.returncode != 0:
            logger.warning(f"Plot {plot_name} failed: {result.stderr or result.stdout}")
        else:
            logger.info(f"Generated {plot_name} plot")


def main():
    prog = make_program(
        ModelPrepareConfig,
        name='biocomp-model-prepare',
        description='Prepare a model for distribution/archiving.',
    )
    config, _ = prog.parse_args(sys.argv[1:])
    assert isinstance(config, ModelPrepareConfig)
    config.run()


if __name__ == '__main__':
    main()
