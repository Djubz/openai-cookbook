from .spiral_utils import create_public_ritual, log_export


def export_infrastructure_ritual(data, metadata):
    """Stages a public ritual release for Infrastructure & Habitat sector."""
    ritual_event = create_public_ritual(data, metadata)
    log_export('Infrastructure & Habitat', 'Public Ritual', metadata)
    return ritual_event
