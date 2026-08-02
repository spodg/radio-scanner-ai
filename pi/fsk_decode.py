"""
FSK data burst decoder for scanner audio.

Demodulates common digital signaling protocols found on VHF/UHF radio:
  - POCSAG: Paging protocol (512/1200/2400 baud, sync 0x7CD215D8)
  - MDC-1200: Motorola/GE PTT-ID (1200 baud AFSK, 1200/1800 Hz)
  - FleetSync: Kenwood unit ID (1200 baud AFSK, similar tones)
"""

import numpy as np


# =============================================================================
# POCSAG Protocol
# =============================================================================

POCSAG_SYNC = 0x7CD215D8
POCSAG_IDLE = 0x7A89C197
POCSAG_BAUDS = [512, 1200, 2400]

_POCSAG_NUMERIC = "0123456789*U -)(."


def _pocsag_decode_alpha(msg_bits):
    """Decode alpha characters from concatenated 20-bit message fields."""
    chars = []
    for i in range(0, len(msg_bits) - 6, 7):
        val = 0
        for j in range(7):
            val = (val << 1) | msg_bits[i + j]
        # POCSAG alpha is bit-reversed 7-bit ASCII
        val = int('{:07b}'.format(val)[::-1], 2)
        if 32 <= val < 127:
            chars.append(chr(val))
        elif val == 0:
            break
    return "".join(chars).strip()


def _pocsag_decode_numeric(msg_bits):
    """Decode numeric characters from concatenated 20-bit message fields."""
    chars = []
    for i in range(0, len(msg_bits) - 3, 4):
        val = 0
        for j in range(4):
            val = (val << 1) | msg_bits[i + j]
        val = int('{:04b}'.format(val)[::-1], 2)
        if val < len(_POCSAG_NUMERIC):
            chars.append(_POCSAG_NUMERIC[val])
    return "".join(chars).strip()


def _decode_pocsag(audio, sr):
    """Attempt POCSAG decode at 512, 1200, and 2400 baud."""
    filtered = _bandpass(audio, sr, 600, 3000)
    results = []
    for baud in POCSAG_BAUDS:
        for invert in (False, True):
            if invert:
                bits = _fsk_demod(filtered, sr, 1800, 1200, baud)
            else:
                bits = _fsk_demod(filtered, sr, 1200, 1800, baud)
            pages = _pocsag_extract(bits, baud)
            if pages:
                results.extend(pages)
                return results
    return results


def _pocsag_extract(bits, baud):
    """Find POCSAG sync words and extract pages from the bit stream."""
    n = len(bits)
    sync_bits = [(POCSAG_SYNC >> (31 - i)) & 1 for i in range(32)]
    pages = []

    for pos in range(n - 32):
        errors = sum(bits[pos + i] != sync_bits[i] for i in range(32))
        if errors > 2:
            continue

        batch_start = pos + 32
        if batch_start + 512 > n:
            continue

        address = None
        msg_bits = []
        func = 0

        for cw_idx in range(16):
            cw_start = batch_start + cw_idx * 32
            if cw_start + 32 > n:
                break
            cw = 0
            for i in range(32):
                cw = (cw << 1) | bits[cw_start + i]

            if cw == POCSAG_IDLE or cw == POCSAG_SYNC:
                if address is not None and msg_bits:
                    pages.append(_pocsag_make_page(address, func, msg_bits, baud))
                    address = None
                    msg_bits = []
                continue

            if (cw >> 31) & 1 == 0:
                # Address codeword
                if address is not None and msg_bits:
                    pages.append(_pocsag_make_page(address, func, msg_bits, baud))
                    msg_bits = []
                addr_part = (cw >> 13) & 0x3FFFF
                func = (cw >> 11) & 0x3
                frame_pos = cw_idx // 2
                address = addr_part * 8 + frame_pos
            else:
                # Message codeword
                for i in range(1, 21):
                    msg_bits.append((cw >> (31 - i)) & 1)

        if address is not None:
            pages.append(_pocsag_make_page(address, func, msg_bits, baud))

    return pages


def _pocsag_make_page(address, func, msg_bits, baud):
    """Build a page result dict."""
    msg = ""
    if msg_bits:
        alpha = _pocsag_decode_alpha(msg_bits)
        numeric = _pocsag_decode_numeric(msg_bits)
        if alpha and all(32 <= ord(c) < 127 for c in alpha):
            msg = alpha
        elif numeric:
            msg = numeric

    func_names = {0: "numeric", 1: "tone-only", 2: "alpha", 3: "alpha"}
    return {
        "protocol": "POCSAG",
        "baud": baud,
        "capcode": str(address),
        "function": func_names.get(func, str(func)),
        "message": msg,
        "unit_id": f"cap:{address}",
    }


# MDC-1200 protocol constants
MDC_BAUD = 1200
MDC_FREQ_MARK = 1200   # Hz (logic 1)
MDC_FREQ_SPACE = 1800  # Hz (logic 0)
MDC_PREAMBLE = 0x07    # 00000111 repeated
MDC_SYNC = 0x5555FFFF  # 32-bit sync word (before bit inversion)

# Bit encoding: MDC uses inverted FSK (mark=0, space=1 in some implementations)
# and the packet is: preamble (7+ bytes of 0x07), sync (0x00, 0x80), then data.

def _bandpass(audio, sr, f_low, f_high, order=4):
    """Simple FFT-based bandpass filter."""
    n = len(audio)
    freqs = np.fft.rfftfreq(n, 1/sr)
    spec = np.fft.rfft(audio)
    mask = (freqs >= f_low) & (freqs <= f_high)
    spec[~mask] = 0
    return np.fft.irfft(spec, n)


def _fsk_demod(audio, sr, f_mark, f_space, baud):
    """
    Demodulate binary FSK by correlating with mark/space tones per symbol.
    Returns a binary array (0/1 per symbol period).
    """
    samples_per_bit = int(sr / baud)
    n_bits = len(audio) // samples_per_bit
    bits = []
    for i in range(n_bits):
        seg = audio[i * samples_per_bit:(i + 1) * samples_per_bit]
        t = np.arange(len(seg)) / sr
        # Correlation with mark and space
        mark_corr = abs(np.sum(seg * np.sin(2 * np.pi * f_mark * t)))
        space_corr = abs(np.sum(seg * np.sin(2 * np.pi * f_space * t)))
        bits.append(1 if mark_corr > space_corr else 0)
    return np.array(bits, dtype=np.uint8)


def _bits_to_bytes(bits, msb_first=True):
    """Convert a bit array to bytes."""
    out = []
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for j in range(8):
            if msb_first:
                byte = (byte << 1) | bits[i + j]
            else:
                byte |= bits[i + j] << j
        out.append(byte)
    return bytes(out)


def _find_mdc_sync(bits):
    """
    Find the MDC-1200 sync pattern in the bit stream.
    MDC preamble is alternating 0101... followed by a specific sync word.
    Returns the bit index after the sync, or -1.
    """
    # Look for a run of alternating bits (preamble) of at least 16 bits
    n = len(bits)
    for i in range(n - 48):
        # Check for alternating pattern (0101 or 1010)
        alt_count = 0
        for j in range(min(24, n - i - 1)):
            if bits[i + j] != bits[i + j + 1]:
                alt_count += 1
            else:
                break
        if alt_count >= 14:
            # Found preamble, data starts after it
            return i + alt_count + 1
    return -1


def _decode_mdc1200(audio, sr):
    """
    Attempt MDC-1200 decode. Returns dict with unit_id, op_code, etc. or None.
    Only proceeds if the audio actually contains energy at the MDC tone pair
    (1200 + 1800 Hz) to avoid false matches on non-MDC data.
    """
    # Verify MDC tones are actually present before attempting decode
    n = len(audio)
    spec = np.abs(np.fft.rfft(audio * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1/sr)
    # Check power at 1200 and 1800 Hz (±50 Hz)
    mask_1200 = (freqs >= 1150) & (freqs <= 1250)
    mask_1800 = (freqs >= 1750) & (freqs <= 1850)
    total_power = np.sum(spec[(freqs > 500) & (freqs < 3000)])
    mdc_power = np.sum(spec[mask_1200]) + np.sum(spec[mask_1800])
    # MDC tones should carry a significant fraction of the energy
    if total_power == 0 or mdc_power / total_power < 0.15:
        return None

    # Bandpass around MDC tones (1000-2000 Hz)
    filtered = _bandpass(audio, sr, 900, 2100)
    
    # Try both polarities (mark=1 and mark=0)
    for invert in (False, True):
        if invert:
            bits = _fsk_demod(filtered, sr, MDC_FREQ_SPACE, MDC_FREQ_MARK, MDC_BAUD)
        else:
            bits = _fsk_demod(filtered, sr, MDC_FREQ_MARK, MDC_FREQ_SPACE, MDC_BAUD)
        
        sync_pos = _find_mdc_sync(bits)
        if sync_pos < 0 or sync_pos + 40 > len(bits):
            continue
        
        # MDC-1200 packet after sync: 5 bytes (40 bits)
        # Byte 0: op code
        # Byte 1-2: unit ID (hex)
        # Byte 3: status/extra
        # Byte 4: checksum
        data_bits = bits[sync_pos:sync_pos + 40]
        data = _bits_to_bytes(data_bits)
        if len(data) < 4:
            continue
        
        op_code = data[0]
        unit_id = (data[1] << 8) | data[2]
        status = data[3] if len(data) > 3 else 0
        
        # Basic sanity: unit IDs are typically in a reasonable range
        if unit_id == 0 or unit_id == 0xFFFF:
            continue
        
        # Decode op code
        op_names = {
            0x01: "PTT-ID (pre)", 0x00: "PTT-ID",
            0x80: "Emergency", 0x81: "Emergency ACK",
            0x20: "Status", 0x22: "Message",
            0x30: "Radio check", 0x31: "Radio check ACK",
            0x46: "Stun", 0x47: "Revive",
        }
        op_name = op_names.get(op_code, f"op=0x{op_code:02X}")
        
        return {
            "protocol": "MDC-1200",
            "unit_id": f"{unit_id:04X}",
            "op_code": f"0x{op_code:02X}",
            "op_name": op_name,
            "status": f"0x{status:02X}",
            "raw": data.hex(),
        }
    return None


def _decode_fleetsync(audio, sr):
    """
    Attempt FleetSync decode. Similar AFSK parameters to MDC but different
    framing. Returns dict or None.
    """
    # Same tone pair check as MDC (FleetSync uses similar tones)
    n = len(audio)
    spec = np.abs(np.fft.rfft(audio * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1/sr)
    mask_1200 = (freqs >= 1150) & (freqs <= 1250)
    mask_1800 = (freqs >= 1750) & (freqs <= 1850)
    total_power = np.sum(spec[(freqs > 500) & (freqs < 3000)])
    fs_power = np.sum(spec[mask_1200]) + np.sum(spec[mask_1800])
    if total_power == 0 or fs_power / total_power < 0.15:
        return None

    # FleetSync: 1200 baud, tones at ~1200/1800 Hz
    filtered = _bandpass(audio, sr, 900, 2100)
    bits = _fsk_demod(filtered, sr, 1200, 1800, 1200)
    
    # FleetSync sync pattern: specific bit sequence
    # Look for the ANI (automatic number identification) format:
    # Preamble + sync + fleet/unit data
    sync_pos = _find_mdc_sync(bits)  # similar preamble structure
    if sync_pos < 0 or sync_pos + 64 > len(bits):
        return None
    
    data_bits = bits[sync_pos:sync_pos + 64]
    data = _bits_to_bytes(data_bits)
    if len(data) < 7:
        return None
    
    # FleetSync ANI: fleet (3 digits) + unit (4 digits) encoded in BCD or binary
    # This is a best-effort extraction
    fleet = ((data[0] & 0x0F) * 100 + (data[1] >> 4) * 10 + (data[1] & 0x0F))
    unit = ((data[2] >> 4) * 1000 + (data[2] & 0x0F) * 100 +
            (data[3] >> 4) * 10 + (data[3] & 0x0F))
    
    if fleet == 0 and unit == 0:
        return None
    
    return {
        "protocol": "FleetSync",
        "fleet": f"{fleet:03d}",
        "unit": f"{unit:04d}",
        "unit_id": f"{fleet:03d}-{unit:04d}",
        "raw": data.hex(),
    }


def decode_fsk(audio: np.ndarray, sr: int) -> list:
    """
    Try all known FSK protocols on the audio clip. Returns a list of
    successfully decoded results (may be empty if nothing decodes cleanly).
    """
    if audio is None or audio.size == 0:
        return []
    
    # Only process clips with enough energy
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 0.01:
        return []
    
    results = []
    
    # Try POCSAG (paging - longer bursts with sync word)
    try:
        r = _decode_pocsag(audio, sr)
        if r:
            results.extend(r)
    except Exception:
        pass

    # Try MDC-1200
    try:
        r = _decode_mdc1200(audio, sr)
        if r:
            results.append(r)
    except Exception:
        pass
    
    # Try FleetSync
    try:
        r = _decode_fleetsync(audio, sr)
        if r:
            results.append(r)
    except Exception:
        pass
    
    return results


def format_results(results: list) -> str:
    """Format decoded FSK results into a human-readable string."""
    if not results:
        return ""
    parts = []
    for r in results:
        proto = r.get("protocol", "?")
        if proto == "POCSAG":
            msg = r.get("message", "")
            cap = r.get("capcode", "?")
            if msg:
                parts.append(f"POCSAG cap:{cap} [{r.get('function','')}] \"{msg}\"")
            else:
                parts.append(f"POCSAG cap:{cap} [{r.get('function','')}]")
        elif proto == "MDC-1200":
            parts.append(f"MDC-1200 ID:{r['unit_id']} {r['op_name']}")
        elif proto == "FleetSync":
            parts.append(f"FleetSync {r['unit_id']}")
        else:
            parts.append(f"{proto} {r.get('unit_id','')}")
    return "; ".join(parts)
