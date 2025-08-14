from .spiral_utils import build_web_archive, log_export


def export_education_web(data, metadata):
    """Publishes Education & Knowledge sector as an interactive web archive."""
    web_archive = build_web_archive(data, metadata)
    log_export('Education & Knowledge', 'Web Archive', metadata)
    return web_archive
