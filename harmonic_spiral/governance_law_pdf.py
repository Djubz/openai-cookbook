from .spiral_utils import create_pdf, log_export


def export_governance_pdf(data, metadata):
    """Exports Governance & Law sector data to PDF, with full spiral metadata."""
    pdf = create_pdf(data, metadata)
    log_export('Governance & Law', 'PDF', metadata)
    return pdf
