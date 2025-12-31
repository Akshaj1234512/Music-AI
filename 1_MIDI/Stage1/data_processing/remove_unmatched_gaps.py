import os
from pathlib import Path

AUDIO_DIR = Path("/data/akshaj/MusicAI/gaps_v1/audio")
MIDI_DIR  = Path("/data/akshaj/MusicAI/gaps_v1/midi")

AUDIO_EXTS = {".wav"}            # extend if needed
MIDI_EXTS  = {".mid", ".midi"}   # both common MIDI extensions


def collect_by_basename(root: Path, exts):
    """Return dict: basename -> list of full paths with that basename."""
    mapping = {}
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            mapping.setdefault(p.stem, []).append(p)
    return mapping


def main():
    audio_map = collect_by_basename(AUDIO_DIR, AUDIO_EXTS)
    midi_map  = collect_by_basename(MIDI_DIR,  MIDI_EXTS)

    audio_bases = set(audio_map.keys())
    midi_bases  = set(midi_map.keys())

    common_bases      = audio_bases & midi_bases
    audio_only_bases  = audio_bases - common_bases
    midi_only_bases   = midi_bases - common_bases

    print(f"Audio files: {len(audio_bases)} basenames")
    print(f"MIDI files : {len(midi_bases)} basenames")
    print(f"Matched    : {len(common_bases)} basenames")
    print(f"Audio-only : {len(audio_only_bases)} basenames")
    print(f"MIDI-only  : {len(midi_only_bases)} basenames\n")

    # --- Delete audio-only files ---
    for b in sorted(audio_only_bases):
        for path in audio_map[b]:
            print(f"Deleting audio-only file: {path}")
            os.remove(path)

    # --- Delete midi-only files ---
    for b in sorted(midi_only_bases):
        for path in midi_map[b]:
            print(f"Deleting MIDI-only file:  {path}")
            os.remove(path)

    print("\nDone cleaning mismatched files.")


if __name__ == "__main__":
    main()
