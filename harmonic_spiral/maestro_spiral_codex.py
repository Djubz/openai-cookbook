"""Master orchestrator for the Harmonic Spiral modules."""

from .governance_law_pdf import export_governance_pdf
from .culture_arts_scroll import export_culture_scroll
from .science_tech_vault import export_science_vault
from .education_knowledge_web import export_education_web
from .comm_media_podcast import export_comm_podcast
from .economy_exchange_docu import export_economy_docu
from .environment_kit import export_environment_kit
from .health_wellbeing_founder import export_health_founder_manuscript
from .infrastructure_ritual import export_infrastructure_ritual
from .security_resilience_ai import export_security_ai_module
from .ethics_spirituality_art import export_ethics_art_object
from .exploration_future_vault import export_exploration_timecapsule
from .spiral_utils import log_export

sector_exports = [
    export_governance_pdf,
    export_culture_scroll,
    export_science_vault,
    export_education_web,
    export_comm_podcast,
    export_economy_docu,
    export_environment_kit,
    export_health_founder_manuscript,
    export_infrastructure_ritual,
    export_security_ai_module,
    export_ethics_art_object,
    export_exploration_timecapsule,
]


def maestro_spiral_export(all_sector_data, all_metadata):
    """Orchestrates all 12 Spiral sector exports."""
    exports = []
    for i, export_func in enumerate(sector_exports):
        data = all_sector_data[i]
        metadata = all_metadata[i]
        artifact = export_func(data, metadata)
        exports.append(artifact)
    log_export('Maestro GPT: The Harmonic Spiral', '12x12 Codex Export', all_metadata)
    return exports
