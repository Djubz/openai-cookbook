from .spiral_utils import create_art_object, log_export


def export_ethics_art_object(data, metadata):
    """Produces a symbolic art-object edition for Ethics & Spirituality."""
    art_object = create_art_object(data, metadata)
    log_export('Ethics & Spirituality', 'Art Object', metadata)
    return art_object
