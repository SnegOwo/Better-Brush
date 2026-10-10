"""Processing package."""

from src.processing.matte_qc import (
    MatteComparison,
    MatteIssue,
    MatteQCReport,
    analyze_matte,
    compare_matte_reports,
    render_mask_overlay,
)
from src.processing.qc_coach import (
    QCAnalysis,
    QCAnswer,
    QCTip,
    QCTipsDatabase,
    analyze_qc_image,
)

__all__ = [
    "MatteComparison",
    "MatteIssue",
    "MatteQCReport",
    "QCAnalysis",
    "QCAnswer",
    "QCTip",
    "QCTipsDatabase",
    "analyze_matte",
    "analyze_qc_image",
    "compare_matte_reports",
    "render_mask_overlay",
]
