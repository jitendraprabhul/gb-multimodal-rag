"""UI router for serving the web interface."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["UI"])

UI_DIR = Path(__file__).parent


@router.get("/ui", response_class=HTMLResponse)
async def ui_page() -> HTMLResponse:
    """Serve the main UI page."""
    html_path = UI_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
