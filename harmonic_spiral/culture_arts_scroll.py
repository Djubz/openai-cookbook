from .spiral_utils import create_scroll_artifact, log_export


def export_culture_scroll(data, metadata):
    """Creates ceremonial codex scroll for Culture & Arts."""
    scroll = create_scroll_artifact(data, metadata)
    log_export('Culture & Arts', 'Scroll', metadata)
    return scroll
