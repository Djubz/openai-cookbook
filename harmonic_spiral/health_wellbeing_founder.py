from .spiral_utils import create_encrypted_manuscript, log_export


def export_health_founder_manuscript(data, metadata):
    """Creates an encrypted Founders' Manuscript for Health & Wellbeing."""
    manuscript = create_encrypted_manuscript(data, metadata)
    log_export('Health & Wellbeing', 'Founder Manuscript', metadata)
    return manuscript
