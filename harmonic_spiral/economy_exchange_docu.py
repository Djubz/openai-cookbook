from .spiral_utils import create_animated_docu, log_export


def export_economy_docu(data, metadata):
    """Renders Economy & Exchange sector as an animated spiral documentary."""
    documentary = create_animated_docu(data, metadata)
    log_export('Economy & Exchange', 'Documentary', metadata)
    return documentary
