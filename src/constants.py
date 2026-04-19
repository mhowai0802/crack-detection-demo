"""Torch-free constants shared between the backend and frontend.

Kept free of heavy imports (``torch``, ``cv2``, ...) so the Streamlit
frontend can reuse class labels and indices without pulling the ML
runtime into its dependency set.
"""

from __future__ import annotations

CLASS_NAMES: list[str] = ["No Crack", "Crack"]

NO_CRACK_INDEX: int = CLASS_NAMES.index("No Crack")
CRACK_INDEX: int = CLASS_NAMES.index("Crack")
