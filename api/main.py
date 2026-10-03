"""Run: python -m uvicorn api.main:app --reload"""

from contextlib import asynccontextmanager

import json
import logging
import shutil

from pathlib import Path
from threading import Lock

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)

from pydantic import BaseModel

from orchestration import (
    Orchestrator,
    ProfilingRequest,
)

from orchestration.errors import (
    NoEligibleAgentError,
)

from registry import AgentRegistry

from .runtime import (
    CONFIG_PATHS,
    ProcessAgent,
)


log = logging.getLogger(__name__)


# ============================================================
# Project paths
# ============================================================

ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

REPORT_DIR = (
    ROOT_DIR
    / "reports"
)

MODEL_UPLOAD_DIR = (
    ROOT_DIR
    / "models"
    / "uploads"
)

MANIFEST_UPLOAD_DIR = (
    ROOT_DIR
    / "manifests"
    / "uploads"
)


REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MANIFEST_UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Request schema
# ============================================================

class ProfileBody(BaseModel):
    manifest_path: str

    backend: str
    device: str

    action: str = "profile"

    report_path: (
        str | None
    ) = None


# ============================================================
# App factory
# ============================================================

def create_app(
    agent_factory=ProcessAgent,
    config_paths=CONFIG_PATHS,
):

    @asynccontextmanager
    async def lifespan(app):
        registry = (
            AgentRegistry()
        )

        app.state.registry = (
            registry
        )

        app.state.orchestrator = (
            Orchestrator(
                registry
            )
        )

        app.state.lock = (
            Lock()
        )

        agents = []

        try:
            for config in config_paths:
                agent = (
                    agent_factory(
                        config
                    )
                )

                agents.append(
                    agent
                )

                try:
                    agent.start()

                except Exception:
                    # GPU unavailable must not
                    # prevent CPU agents.
                    log.exception(
                        "Agent startup failed: %s",
                        config,
                    )

                registry.register_agent(
                    agent
                )

            yield

        finally:
            for agent in reversed(
                agents
            ):
                try:
                    agent.stop()

                except Exception:
                    log.exception(
                        "Agent shutdown failed"
                    )

            registry.clear()


    app = FastAPI(
        title=(
            "Model Profiling API"
        ),
        lifespan=lifespan,
    )


    # ========================================================
    # Health
    # ========================================================

    @app.get("/health")
    def health():
        return {
            "status": "ok",
        }


    # ========================================================
    # Agents
    # ========================================================

    @app.get("/agents")
    def agents():
        with app.state.lock:
            return [
                record.to_dict()
                for record
                in (
                    app.state
                    .registry
                    .list_agents()
                )
            ]


    # ========================================================
    # Upload model
    # ========================================================

    @app.post("/upload/model")
    async def upload_model(
        file: UploadFile = File(...)
    ):
        filename = Path(
            file.filename or ""
        ).name

        if not filename:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid model filename"
                ),
            )


        extension = (
            Path(filename)
            .suffix
            .lower()
        )


        allowed_extensions = {
            ".onnx",
            ".pth",
            ".pt",
        }


        if (
            extension
            not in allowed_extensions
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported model "
                    f"format: {extension}"
                ),
            )


        target = (
            MODEL_UPLOAD_DIR
            / filename
        )


        try:
            with target.open(
                "wb"
            ) as output:
                shutil.copyfileobj(
                    file.file,
                    output,
                )

        except OSError as error:
            log.exception(
                "Could not save uploaded model"
            )

            raise HTTPException(
                status_code=500,
                detail=str(error),
            ) from error

        finally:
            await file.close()


        relative_path = (
            target.relative_to(
                ROOT_DIR
            )
        )


        return {
            "type":
                "model",

            "filename":
                filename,

            "path":
                str(
                    relative_path
                ).replace(
                    "\\",
                    "/",
                ),

            "absolute_path":
                str(target),
        }


    # ========================================================
    # Upload manifest YAML
    # ========================================================

    @app.post("/upload/manifest")
    async def upload_manifest(
        file: UploadFile = File(...)
    ):
        filename = Path(
            file.filename or ""
        ).name

        if not filename:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid manifest filename"
                ),
            )


        extension = (
            Path(filename)
            .suffix
            .lower()
        )


        if extension not in {
            ".yaml",
            ".yml",
        }:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Manifest must be "
                    ".yaml or .yml"
                ),
            )


        target = (
            MANIFEST_UPLOAD_DIR
            / filename
        )


        try:
            with target.open(
                "wb"
            ) as output:
                shutil.copyfileobj(
                    file.file,
                    output,
                )

        except OSError as error:
            log.exception(
                "Could not save uploaded manifest"
            )

            raise HTTPException(
                status_code=500,
                detail=str(error),
            ) from error

        finally:
            await file.close()


        relative_path = (
            target.relative_to(
                ROOT_DIR
            )
        )


        return {
            "type":
                "manifest",

            "filename":
                filename,

            "path":
                str(
                    relative_path
                ).replace(
                    "\\",
                    "/",
                ),

            "absolute_path":
                str(target),
        }


    # ========================================================
    # Profiling
    # ========================================================

    @app.post("/profile")
    def profile(
        body: ProfileBody
    ):
        try:
            request = (
                ProfilingRequest(
                    **body.model_dump()
                )
            )


            # Existing Orchestrator is
            # synchronous.
            with app.state.lock:
                result = (
                    app.state
                    .orchestrator
                    .execute(
                        request
                    )
                    .to_dict()
                )


            return result


        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=str(error),
            ) from error


        except (
            NoEligibleAgentError
        ) as error:
            raise HTTPException(
                status_code=503,
                detail=str(error),
            ) from error


        except Exception as error:
            log.exception(
                "Profiling request failed"
            )

            raise HTTPException(
                status_code=500,
                detail=str(error),
            ) from error


    # ========================================================
    # Reports - list
    # ========================================================

    @app.get("/reports")
    def list_reports():
        reports = []


        for path in sorted(
            REPORT_DIR.glob(
                "*.json"
            ),
            key=lambda p:
                p.stat().st_mtime,
            reverse=True,
        ):
            try:
                stat = (
                    path.stat()
                )


                reports.append({
                    "filename":
                        path.name,

                    "path":
                        str(path),

                    "size_bytes":
                        stat.st_size,

                    "modified_at":
                        stat.st_mtime,
                })


            except OSError:
                continue


        return reports


    # ========================================================
    # Reports - detail
    # ========================================================

    @app.get(
        "/reports/{filename}"
    )
    def get_report(
        filename: str
    ):
        # Prevent ../ traversal.
        safe_name = (
            Path(filename)
            .name
        )


        if safe_name != filename:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid report filename"
                ),
            )


        if not (
            safe_name
            .lower()
            .endswith(".json")
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Report must be "
                    "a JSON file"
                ),
            )


        report_path = (
            REPORT_DIR
            / safe_name
        )


        if not (
            report_path.exists()
        ):
            raise HTTPException(
                status_code=404,
                detail=(
                    "Report not found"
                ),
            )


        if not (
            report_path.is_file()
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid report"
                ),
            )


        try:
            with report_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                return (
                    json.load(file)
                )


        except (
            json.JSONDecodeError
        ) as error:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Report file is "
                    "not valid JSON"
                ),
            ) from error


        except OSError as error:
            raise HTTPException(
                status_code=500,
                detail=str(error),
            ) from error


    return app


app = create_app()