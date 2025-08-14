from .spiral_utils import create_stakeholder_kit, log_export


def export_environment_kit(data, metadata):
    """Generates a Global Stakeholder Kit for Environment & Sustainability."""
    kit = create_stakeholder_kit(data, metadata)
    log_export('Environment & Sustainability', 'Stakeholder Kit', metadata)
    return kit
