"""TPClass: A Two-Phase Predictor for Membrane Transporter Annotation.

TPClass is an ESM-2-based two-phase predictor for membrane transporter
identification and TC (Transporter Classification) family assignment.
It supports substrate-specific transporter mining via custom reference
sets, with validated case studies for HMO, terpenoid (squalene), and
isobutanol exporter discovery.

The package mirrors the released pipeline: the same ESM-2 pooling masks, the
same Phase 1 / Phase 2 architectures, the same ensemble weights and the same
cascade semantics.
"""
__version__ = "1.0.0"

from tpclass.core.predictor import TPClassPredictor
from tpclass.core.phase1 import Phase1Classifier, TransportClassifier
from tpclass.core.phase2 import Phase2Retriever
from tpclass.filters.substrate import SubstrateFilter
from tpclass.filters.three_layer import ThreeLayerFilter
from tpclass.analysis.esm_vs_blast import compare_esm_vs_blast
from tpclass.analysis.compare_reports import compare_reports, format_report
from tpclass.analysis.family_metrics import evaluate_family, evaluate_from_files
from tpclass.analysis.paired_tests import compare_models, mcnemar_exact

__all__ = [
    "TPClassPredictor",
    "Phase1Classifier",
    "Phase2Retriever",
    "TransportClassifier",
    "SubstrateFilter",
    "ThreeLayerFilter",
    "compare_esm_vs_blast",
    "compare_reports",
    "format_report",
    "evaluate_family",
    "evaluate_from_files",
    "compare_models",
    "mcnemar_exact",
]
