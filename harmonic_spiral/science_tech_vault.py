from .spiral_utils import encrypt_and_seal, log_export


def export_science_vault(data, metadata):
    """Seals Science & Technology data in a cryptographically protected vault document."""
    vault_doc = encrypt_and_seal(data, metadata)
    log_export('Science & Tech', 'Vault', metadata)
    return vault_doc
