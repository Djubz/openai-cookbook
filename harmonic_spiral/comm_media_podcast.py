from .spiral_utils import create_audio_narrative, log_export


def export_comm_podcast(data, metadata):
    """Produces spiral podcast/audio book from Communication & Media data."""
    audio = create_audio_narrative(data, metadata)
    log_export('Comm & Media', 'Podcast', metadata)
    return audio
