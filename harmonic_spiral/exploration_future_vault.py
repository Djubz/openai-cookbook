from .spiral_utils import create_time_capsule, log_export


def export_exploration_timecapsule(data, metadata):
    """Encodes Exploration & Future sector into a quantum vault time capsule."""
    capsule = create_time_capsule(data, metadata)
    log_export('Exploration & Future', 'Time Capsule', metadata)
    return capsule
