from .spiral_utils import create_ai_module, log_export


def export_security_ai_module(data, metadata):
    """Embeds Security & Resilience as an AI-integrated learning module."""
    ai_module = create_ai_module(data, metadata)
    log_export('Security & Resilience', 'AI Module', metadata)
    return ai_module
