# Nonce tracking to prevent replay attacks
import time
from substrateinterface import Keypair

_NONCE_CACHE: dict[str, int] = {}


def verify_bittensor_request(
    dendrite_hotkey: str,
    dendrite_nonce: str,
    dendrite_signature: str,
    dendrite_uuid: str,
    axon_hotkey: str,
    body_hash: str = "",
    timeout: float = 12.0,
) -> tuple[bool, str]:
    """
    Verify a request from a Bittensor validator (dendrite).
    
    This implements the same verification logic as bittensor.core.axon to ensure
    requests are authentic and not replayed.
    
    Args:
        dendrite_hotkey: The validator's SS58 hotkey address
        dendrite_nonce: Timestamp nonce from the validator (nanoseconds)
        dendrite_signature: Cryptographic signature from the validator
        dendrite_uuid: Unique identifier for the validator
        axon_hotkey: This miner's hotkey (for signature verification)
        body_hash: Optional hash of request body
        timeout: Request timeout for nonce window calculation
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Validate nonce (prevent replay attacks)
        nonce = int(dendrite_nonce)
        current_time = time.time_ns()
        
        # Check if nonce is within acceptable window (4 seconds + timeout)
        # Manual calculation to avoid float precision issues
        ALLOWED_DELTA = 4_000_000_000  # 4 seconds in nanoseconds
        NANOSECONDS_IN_SECOND = 1_000_000_000
        timeout_ns = int(timeout * NANOSECONDS_IN_SECOND)
        allowed_window = current_time - ALLOWED_DELTA - timeout_ns
        
        if nonce < allowed_window:
            return False, f"Nonce too old (replay attack?): nonce age={(current_time - nonce) / NANOSECONDS_IN_SECOND:.1f}s > allowed={((ALLOWED_DELTA + timeout_ns) / NANOSECONDS_IN_SECOND):.1f}s"
        
        # Check if nonce is too far in the future (max 3 seconds ahead)
        if nonce > current_time + 3_000_000_000:
            return False, f"Nonce too far in future: {nonce} > {current_time + 3_000_000_000}"
        
        # Check for duplicate nonce from this hotkey (simple replay prevention)
        cache_key = f"{dendrite_hotkey}:{nonce}"
        if cache_key in _NONCE_CACHE:
            return False, "Duplicate nonce (replay attack?)"
        
        # Verify cryptographic signature
        # Message format: {nonce}.{dendrite_hotkey}.{axon_hotkey}.{uuid}.{body_hash}
        message = f"{nonce}.{dendrite_hotkey}.{axon_hotkey}.{dendrite_uuid}.{body_hash}"
        
        # Remove '0x' prefix if present
        sig_hex = dendrite_signature[2:] if dendrite_signature.startswith('0x') else dendrite_signature
        
        try:
            signature_bytes = bytes.fromhex(sig_hex)
        except ValueError:
            return False, "Invalid signature format"
        
        # Verify signature using the dendrite's (validator's) public key
        try:
            keypair = Keypair(ss58_address=dendrite_hotkey)
            is_valid = keypair.verify(message, signature_bytes)
            
            if not is_valid:
                return False, "Signature verification failed"
                
        except Exception as e:
            return False, f"Signature verification error: {str(e)}"
        
        # Cache the nonce to prevent replay (cleanup old entries periodically)
        _NONCE_CACHE[cache_key] = current_time
        
        # Cleanup old nonces (keep last 1000 entries)
        if len(_NONCE_CACHE) > 1000:
            # Remove oldest 200 entries
            sorted_keys = sorted(_NONCE_CACHE.items(), key=lambda x: x[1])
            for key, _ in sorted_keys[:200]:
                del _NONCE_CACHE[key]
        
        return True, ""
        
    except ValueError as e:
        return False, f"Invalid nonce format: {str(e)}"
    except Exception as e:
        return False, f"Verification error: {str(e)}"