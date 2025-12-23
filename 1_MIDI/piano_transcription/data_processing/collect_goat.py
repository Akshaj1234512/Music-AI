from pathlib import Path
import shutil
from typing import Optional


GOAT_ROOT = Path("/data/shamakg/GOAT")
DATA_DIR  = GOAT_ROOT / "data"

OUT_AUDIO = Path("/data/akshaj/MusicAI/GOAT/audio")
OUT_MIDI  = Path("/data/akshaj/MusicAI/GOAT/midi")

AMP_NUMS = {1, 2, 3, 4, 5}
MIDI_EXTS = {".mid", ".midi"}


def is_amp_wav(p: Path) -> bool:
    if p.suffix.lower() != ".wav":
        return False
    name = p.stem
    if "_amp_" not in name:
        return False
    try:
        amp = int(name.rsplit("_amp_", 1)[1])
    except ValueError:
        return False
    return amp in AMP_NUMS


def is_fine_aligned_midi(p: Path) -> bool:
    if p.suffix.lower() not in MIDI_EXTS:
        return False
    return p.stem.endswith("_fine_aligned")


def is_regular_midi(p: Path) -> bool:
    """A MIDI that is NOT the fine-aligned one."""
    if p.suffix.lower() not in MIDI_EXTS:
        return False
    return not p.stem.endswith("_fine_aligned")


def safe_copy(src: Path, dst: Path) -> str:
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return "copied"

    if src.stat().st_size == dst.stat().st_size:
        return "skipped"

    return "conflict"


def pick_midi_file(item_dir: Path, item_name: str) -> Optional[Path]:

    """
    Prefer *_fine_aligned.mid/.midi.
    If missing, fall back to a regular MIDI:
      - first try exact name: {item_name}.mid or {item_name}.midi
      - else pick any regular MIDI in the folder (sorted)
    """
    # 1) fine-aligned
    fine = sorted([p for p in item_dir.iterdir() if p.is_file() and is_fine_aligned_midi(p)])
    if fine:
        return fine[0]

    # 2) exact regular fallback: item_165.mid / item_165.midi
    for ext in [".mid", ".midi"]:
        candidate = item_dir / f"{item_name}{ext}"
        if candidate.exists() and candidate.is_file():
            return candidate

    # 3) any regular MIDI
    regular = sorted([p for p in item_dir.iterdir() if p.is_file() and is_regular_midi(p)])
    if regular:
        return regular[0]

    return None


def main():
    if not DATA_DIR.exists():
        raise SystemExit(f"Missing GOAT data dir: {DATA_DIR}")

    OUT_AUDIO.mkdir(parents=True, exist_ok=True)
    OUT_MIDI.mkdir(parents=True, exist_ok=True)

    audio_copied = audio_skipped = audio_conflicts = 0
    midi_copied = midi_skipped = midi_conflicts = 0
    items_seen = 0
    items_missing_midi = 0
    items_used_regular = 0

    for item_dir in sorted(DATA_DIR.iterdir()):
        if not item_dir.is_dir():
            continue

        items_seen += 1
        item_name = item_dir.name  # e.g., item_165

        # amp wavs
        amp_wavs = sorted([p for p in item_dir.iterdir() if p.is_file() and is_amp_wav(p)])
        for wav in amp_wavs:
            out_name = f"{item_name}__{wav.name}"
            status = safe_copy(wav, OUT_AUDIO / out_name)
            if status == "copied":
                audio_copied += 1
            elif status == "skipped":
                audio_skipped += 1
            else:
                audio_conflicts += 1
                print(f"[AUDIO CONFLICT] {wav} -> {OUT_AUDIO / out_name}")

        # MIDI (fine-aligned preferred, else regular)
        midi_file = pick_midi_file(item_dir, item_name)
        if midi_file is None:
            items_missing_midi += 1
        else:
            if is_regular_midi(midi_file) and not is_fine_aligned_midi(midi_file):
                # this means we fell back (could also be regular even if fine missing)
                if not midi_file.stem.endswith("_fine_aligned"):
                    # Count only if fine-aligned was absent (pick_midi_file implies that)
                    items_used_regular += 1

            out_name = f"{item_name}__{midi_file.name}"
            status = safe_copy(midi_file, OUT_MIDI / out_name)
            if status == "copied":
                midi_copied += 1
            elif status == "skipped":
                midi_skipped += 1
            else:
                midi_conflicts += 1
                print(f"[MIDI CONFLICT] {midi_file} -> {OUT_MIDI / out_name}")

    print("\n=== DONE ===")
    print(f"Items scanned: {items_seen}")
    print(f"Items missing ANY MIDI: {items_missing_midi}")
    print(f"Items used REGULAR fallback (no fine_aligned): {items_used_regular}")
    print(f"\nAudio: copied={audio_copied}, skipped={audio_skipped}, conflicts={audio_conflicts}")
    print(f"MIDI : copied={midi_copied}, skipped={midi_skipped}, conflicts={midi_conflicts}")
    print(f"\nOutput dirs:\n  {OUT_AUDIO}\n  {OUT_MIDI}")


if __name__ == "__main__":
    main()
