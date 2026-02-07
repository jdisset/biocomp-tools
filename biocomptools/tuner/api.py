"""biocomp-tuner: Interactive design parameter explorer with web UI."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Annotated, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from dracon.commandline import Arg, dracon_program
from dracon.diagnostics import DraconError, handle_dracon_error

from biocomp.design_targets import DataTarget, SVGTarget, TargetUnion
from biocomp.network import Network, recipe_to_networks
from biocomp.recipe import Recipe
from biocomptools.logging_config import get_logger, setup_logging
from biocomptools.modelmodel import BiocompModel
from biocomptools.toollib.common import config
from biocomptools.toollib.modelselector import ModelSelector

from .param_schema import extract_editable_params, extract_grouped_params, group_params_by_category
from .session import TunerSession

logger = get_logger(__name__)

TUNER_TYPES = [Recipe, Network, SVGTarget, DataTarget, BiocompModel, ModelSelector]


@dracon_program(
    name="biocomp-tuner",
    description="Interactive web tool for exploring design parameter space.",
    context_types=TUNER_TYPES,
    context={"BIOCOMP_ROOT": Path(config.paths.root).expanduser().resolve()},
)
class TunerProgram(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    scaffold: Annotated[Optional[Recipe | list[Recipe]], Arg(help="Scaffold recipe(s) to tune")] = (
        None
    )
    network: Annotated[
        Optional[Network | list[Network]], Arg(help="Pre-built network(s) to tune")
    ] = None
    target: Annotated[TargetUnion | list[TargetUnion], Arg(help="Target pattern")] = Field(
        default_factory=list
    )
    model_name: Annotated[Optional[str], Arg(help="Model path or signature")] = os.environ.get(
        "BIOCOMP_DESIGNER_MODEL"
    )
    model_selector: Annotated[
        Optional[ModelSelector], Arg(help="Model selector for complex queries")
    ] = None
    grid_resolution: Annotated[tuple[int, int], Arg(help="Grid resolution")] = (32, 32)
    w_sinkhorn: Annotated[float, Arg(help="Sinkhorn loss weight")] = 1.0
    w_lncc: Annotated[float, Arg(help="LNCC loss weight")] = 0.5
    w_mse: Annotated[float, Arg(help="MSE loss weight")] = 1.0
    w_simse: Annotated[float, Arg(help="SIMSE loss weight")] = 1.0
    host: Annotated[str, Arg(help="Host to bind to")] = "127.0.0.1"
    port: Annotated[int, Arg(help="Port to bind to")] = 8765

    def model_post_init(self, __context):
        self._model: Optional[BiocompModel] = None
        self._session: Optional[TunerSession] = None
        self._network: Optional[Network] = None
        self._target: Optional[TargetUnion] = None

    def _load_model(self) -> BiocompModel:
        if self.model_name and self.model_name.strip():
            model_path = Path(self.model_name)
            if model_path.exists() and model_path.suffix == ".pickle":
                logger.info(f"Loading model from path: {model_path}")
                return BiocompModel.load(model_path)
            else:
                from biocomptools.toollib.common import get_db_engine
                from sqlmodel import Session

                effective_selector = ModelSelector(name=self.model_name)
                engine = get_db_engine()
                with Session(engine) as session:
                    selected_model = effective_selector.get_model(session)
                    assert selected_model is not None, f"Model not found: {self.model_name}"
                    logger.info(f"Loading model: {selected_model.name}")
                    model = selected_model.load()
                    session.expunge_all()
                return model

        if self.model_selector:
            from biocomptools.toollib.common import get_db_engine
            from sqlmodel import Session

            engine = get_db_engine()
            with Session(engine) as session:
                selected_model = self.model_selector.get_model(session)
                assert selected_model is not None, "No model found for selector"
                logger.info(f"Loading model: {selected_model.name}")
                model = selected_model.load()
                session.expunge_all()
            return model

        raise ValueError("Either model_name or model_selector must be provided")

    def _resolve_network(self) -> Network:
        if self.network is not None:
            return self.network[0] if isinstance(self.network, list) else self.network

        if self.scaffold is not None:
            scaffold = self.scaffold[0] if isinstance(self.scaffold, list) else self.scaffold
            networks = recipe_to_networks(scaffold, invert=True, inversion_mode="main")
            assert len(networks) > 0, "No networks built from scaffold"
            return networks[0]

        raise ValueError("Either network or scaffold must be provided")

    def _resolve_target(self) -> TargetUnion:
        if isinstance(self.target, list):
            assert len(self.target) > 0, "No targets provided"
            return self.target[0]
        return self.target

    def initialize_session(self) -> TunerSession:
        self._model = self._load_model()
        self._network = self._resolve_network()
        self._target = self._resolve_target()

        logger.info(f"Network: {self._network.name}")
        logger.info(f"Target: {getattr(self._target, 'name', 'unnamed')}")
        logger.info(
            f"Loss weights: sinkhorn={self.w_sinkhorn}, lncc={self.w_lncc}, mse={self.w_mse}, simse={self.w_simse}"
        )

        self._session = TunerSession(
            model=self._model,
            network=self._network,
            target=self._target,
            grid_resolution=self.grid_resolution,
            w_sinkhorn=self.w_sinkhorn,
            w_lncc=self.w_lncc,
            w_mse=self.w_mse,
            w_simse=self.w_simse,
        )
        self._session.initialize()
        return self._session

    def run(self):
        import uvicorn

        self.initialize_session()
        assert self._session is not None

        app = create_tuner_app(self._session, self)
        logger.info(f"Starting biocomp-tuner at http://{self.host}:{self.port}")
        uvicorn.run(app, host=self.host, port=self.port)


TUNER_APP_DIR = Path(__file__).parent.parent / "tuner_app_react"
_current_session: Optional[TunerSession] = None
_current_program: Optional[TunerProgram] = None


class ParamUpdateRequest(BaseModel):
    updates: dict[str, list | float]


def create_tuner_app(session: TunerSession, program: TunerProgram) -> FastAPI:
    global _current_session, _current_program
    _current_session = session
    _current_program = program

    app = FastAPI(title="biocomp-tuner", description="Interactive design parameter explorer")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if TUNER_APP_DIR.exists():
        assets_dir = TUNER_APP_DIR / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/")
    async def serve_index():
        index_path = TUNER_APP_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return HTMLResponse("<h1>biocomp-tuner</h1><p>Frontend not found.</p>")

    @app.get("/status")
    async def get_status():
        scaffold_name = None
        if program.scaffold:
            s = program.scaffold[0] if isinstance(program.scaffold, list) else program.scaffold
            scaffold_name = s.name

        target_name = None
        if program.target:
            t = program.target[0] if isinstance(program.target, list) else program.target
            target_name = getattr(t, "name", None)

        model_display = Path(program.model_name).stem if program.model_name else None

        return {
            "initialized": _current_session is not None,
            "scaffold_name": scaffold_name,
            "target_name": target_name,
            "model_name": model_display,
            "network_name": _current_session.network.name if _current_session else None,
            "grid_resolution": program.grid_resolution,
        }

    @app.get("/params")
    async def get_params():
        if _current_session is None:
            raise HTTPException(status_code=400, detail="Session not initialized")
        try:
            descriptors = extract_editable_params(_current_session)
            grouped = group_params_by_category(descriptors)
            return {
                "params": [d.to_dict() for d in descriptors],
                "by_category": {
                    cat: [d.to_dict() for d in params] for cat, params in grouped.items()
                },
                "total_count": len(descriptors),
            }
        except Exception as e:
            logger.exception("Failed to get params")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/params")
    async def update_params(request: ParamUpdateRequest):
        if _current_session is None:
            raise HTTPException(status_code=400, detail="Session not initialized")
        try:
            for path, value in request.updates.items():
                _current_session.set_param(path, value)
            return {"status": "ok", "updated": list(request.updates.keys())}
        except Exception as e:
            logger.exception("Failed to update params")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/compute")
    async def compute():
        if _current_session is None:
            raise HTTPException(status_code=400, detail="Session not initialized")
        try:
            result = _current_session.compute()
            return result.to_dict()
        except Exception as e:
            logger.exception("Failed to compute")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/reset")
    async def reset_params(seed: Optional[int] = None):
        if _current_session is None:
            raise HTTPException(status_code=400, detail="Session not initialized")
        try:
            _current_session.reset_params(seed)
            return {"status": "ok", "seed": seed}
        except Exception as e:
            logger.exception("Failed to reset params")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/export")
    async def export_params():
        if _current_session is None:
            raise HTTPException(status_code=400, detail="Session not initialized")
        try:
            params_json = _current_session.export_params_json()
            return JSONResponse(
                content=json.loads(params_json),
                headers={"Content-Disposition": "attachment; filename=tuner_params.json"},
            )
        except Exception as e:
            logger.exception("Failed to export params")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/import")
    async def import_params(params: dict):
        if _current_session is None:
            raise HTTPException(status_code=400, detail="Session not initialized")
        try:
            _current_session.set_params_dict(params)
            return {"status": "ok", "imported": len(params)}
        except Exception as e:
            logger.exception("Failed to import params")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/config")
    async def get_config():
        """Get current configuration for display."""
        if _current_program is None:
            raise HTTPException(status_code=400, detail="Program not initialized")
        return {
            "model_name": _current_program.model_name,
            "scaffold_name": (
                _current_program.scaffold[0].name
                if isinstance(_current_program.scaffold, list) and _current_program.scaffold
                else getattr(_current_program.scaffold, "name", None)
            ),
            "target_name": (
                getattr(_current_program.target[0], "name", None)
                if isinstance(_current_program.target, list) and _current_program.target
                else getattr(_current_program.target, "name", None)
            ),
            "grid_resolution": _current_program.grid_resolution,
            "loss_weights": {
                "sinkhorn": _current_program.w_sinkhorn,
                "lncc": _current_program.w_lncc,
                "mse": _current_program.w_mse,
                "simse": _current_program.w_simse,
            },
        }

    @app.get("/params/groups")
    async def get_params_groups():
        """Get hierarchical parameter structure for tree view."""
        if _current_session is None:
            raise HTTPException(status_code=400, detail="Session not initialized")
        try:
            groups = extract_grouped_params(_current_session)
            return {
                "groups": [g.to_dict() for g in groups],
                "total_count": sum(len(g.params) for g in groups),
            }
        except Exception as e:
            logger.exception("Failed to get params groups")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/compute/detailed")
    async def compute_detailed(include_contributions: bool = False):
        """Compute with optional per-pixel loss contributions."""
        if _current_session is None:
            raise HTTPException(status_code=400, detail="Session not initialized")
        try:
            result = _current_session.compute(include_contributions=include_contributions)
            return result.to_dict()
        except Exception as e:
            logger.exception("Failed to compute detailed")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/diagram")
    async def get_diagram(plot_type: str = "all"):
        """Get network diagram as SVG."""
        if _current_session is None:
            raise HTTPException(status_code=400, detail="Session not initialized")
        try:
            from biocomptools.circuitplot import CircuitPlotConfig, run_circuitplot

            with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
                output_path = Path(f.name)

            cfg = CircuitPlotConfig(
                recipe=None,
                output=str(output_path),
                plot_type=plot_type,
                invert=False,
            )
            cfg._network = _current_session.network
            run_circuitplot(cfg)
            svg_content = output_path.read_text()
            output_path.unlink(missing_ok=True)
            return {"svg": svg_content, "plot_type": plot_type}
        except Exception as e:
            logger.exception("Failed to generate diagram")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/design/compare")
    async def load_design_for_comparison(run_dir: str):
        """Load a design run for comparison."""
        try:
            run_path = Path(run_dir)
            if not run_path.exists():
                raise HTTPException(status_code=404, detail=f"Design run not found: {run_dir}")

            history_file = run_path / "step_history_data" / "summary.json"
            if history_file.exists():
                import json

                with open(history_file) as f:
                    summary = json.load(f)
                return {
                    "final_loss": summary.get("final_loss"),
                    "loss_history": summary.get("loss_history", [])[-100:],
                }

            return {"error": "Design run summary not found", "run_dir": run_dir}
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Failed to load design run")
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket for real-time parameter updates."""
        await websocket.accept()

        if _current_session is None:
            await websocket.close(code=1008, reason="Session not initialized")
            return

        try:
            while True:
                data = await websocket.receive_json()
                action = data.get("action")

                if action == "update_param":
                    _current_session.set_param(data["path"], data["value"])
                    result = _current_session.compute()
                    await websocket.send_json({"type": "compute_result", "data": result.to_dict()})

                elif action == "update_params_batch":
                    for path, value in data.get("updates", {}).items():
                        _current_session.set_param(path, value)
                    result = _current_session.compute()
                    await websocket.send_json({"type": "compute_result", "data": result.to_dict()})

                elif action == "compute":
                    result = _current_session.compute()
                    await websocket.send_json({"type": "compute_result", "data": result.to_dict()})

                elif action == "reset":
                    _current_session.reset_params(data.get("seed"))
                    result = _current_session.compute()
                    await websocket.send_json({"type": "compute_result", "data": result.to_dict()})

                else:
                    await websocket.send_json(
                        {"type": "error", "message": f"Unknown action: {action}"}
                    )

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")

    return app


def main():
    setup_logging()
    try:
        TunerProgram.cli()
    except DraconError as e:
        handle_dracon_error(e, exit_code=1)
    except Exception as e:
        root = e
        while root.__cause__ is not None:
            root = root.__cause__
        if isinstance(root, DraconError):
            handle_dracon_error(root, exit_code=1)
        raise


if __name__ == "__main__":
    main()
