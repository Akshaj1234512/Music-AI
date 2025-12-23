import os
from pathlib import Path

# You need: pip install music21
from music21 import converter

# Hardcoded paths (your GAPS setup)
XML_DIR = Path("/data/subhashprasad/GAPS_audio/musicxml")
OUT_MIDI_DIR = Path("/data/akshaj/MusicAI/GAPS_aligned")


def convert_one_xml(xml_path: Path, out_dir: Path):
    """
    Convert a single MusicXML file to MIDI and save it in out_dir
    with the same basename but .mid extension.
    """
    try:
        score = converter.parse(str(xml_path))
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / (xml_path.stem + ".mid")

        score.write("midi", fp=str(out_path))
        print(f"[OK]  {xml_path.name} -> {out_path}")
        return True
    except Exception as e:
        print(f"[ERR] {xml_path} -> {e}")
        return False


def main():
    if not XML_DIR.exists():
        print(f"[FATAL] XML directory does not exist: {XML_DIR}")
        return

    print(f"Input XML dir : {XML_DIR}")
    print(f"Output MIDI dir: {OUT_MIDI_DIR}")

    # Collect .xml / .musicxml files (non-recursive; change to rglob if nested)
    xml_files = sorted(
        [p for p in XML_DIR.iterdir()
         if p.is_file() and p.suffix.lower() in [".xml", ".musicxml"]]
    )

    print(f"Found {len(xml_files)} MusicXML files.")

    if not xml_files:
        return

    ok = 0
    err = 0

    for xml_path in xml_files:
        if convert_one_xml(xml_path, OUT_MIDI_DIR):
            ok += 1
        else:
            err += 1

    print("\n=== Summary ===")
    print(f"Total XML files:   {len(xml_files)}")
    print(f"Successfully conv.: {ok}")
    print(f"Failed:            {err}")


if __name__ == "__main__":
    main()
