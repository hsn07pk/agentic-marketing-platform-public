"""
Encryption utilities for secure storage of sensitive configuration values.
Uses Fernet symmetric encryption from cryptography library.
"""
import os
import base64
import hashlib
from typing import Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging

logger = logging.getLogger(__name__)

class ConfigEncryption:
    """
    Handles encryption/decryption of sensitive configuration values.
    Key derived from MASTER_SECRET env var or auto-generated on first run.
    """
    
    _instance: Optional['ConfigEncryption'] = None
    _fernet: Optional[Fernet] = None
    
    # Default master secret for initial bootstrap (should be changed in production)
    DEFAULT_MASTER_SECRET = "CHANGE_ME_IN_PRODUCTION_DO_NOT_USE_DEFAULT"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._fernet is None:
            self._initialize_encryption()
    
    def _initialize_encryption(self):
    
        master_secret = os.environ.get('MASTER_SECRET', self.DEFAULT_MASTER_SECRET)
        
        self._fernet = self._derive_key(master_secret)
        logger.info("Encryption service initialized")
    
    def _derive_key(self, master_secret: str, salt: bytes = None) -> Fernet:
        """Derive a Fernet key from a master secret using PBKDF2."""
        if salt is None:
            # Use a fixed salt for deterministic key derivation
            # In production, you might want to store this separately
            salt = b'dev-only-static-salt-change-me'
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(master_secret.encode()))
        return Fernet(key)
    
    def encrypt(self, plaintext: str) -> str:
        """Encrypt a plaintext string, returns base64-encoded string."""
        if not plaintext:
            return ""
        
        try:
            encrypted = self._fernet.encrypt(plaintext.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise ValueError("Failed to encrypt value") from e
    
    def decrypt(self, ciphertext: str) -> str:
        """Decrypt a base64-encoded encrypted string."""
        if not ciphertext:
            return ""
        
        try:
            encrypted = base64.urlsafe_b64decode(ciphertext.encode())
            decrypted = self._fernet.decrypt(encrypted)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Failed to decrypt value - key may have changed") from e
    
    def is_encrypted(self, value: str) -> bool:
        """Check if a value appears to be encrypted."""
        if not value:
            return False
        
        try:
            decoded = base64.urlsafe_b64decode(value.encode())
            # Fernet tokens start with a specific version byte
            return len(decoded) > 0 and decoded[0] == 0x80
        except Exception:
            return False
    
    def mask_value(self, value: str, show_chars: int = 4) -> str:
        """Mask a sensitive value for display (e.g., "••••••abcd")."""
        if not value:
            return ""
        
        if len(value) <= show_chars:
            return "•" * len(value)
        
        return "•" * (len(value) - show_chars) + value[-show_chars:]

_encryption = None

def get_encryption() -> ConfigEncryption:
    global _encryption
    if _encryption is None:
        _encryption = ConfigEncryption()
    return _encryption

def encrypt_value(plaintext: str) -> str:
    return get_encryption().encrypt(plaintext)

def decrypt_value(ciphertext: str) -> str:
    return get_encryption().decrypt(ciphertext)

def mask_sensitive_value(value: str, show_chars: int = 4) -> str:
    return get_encryption().mask_value(value, show_chars)
