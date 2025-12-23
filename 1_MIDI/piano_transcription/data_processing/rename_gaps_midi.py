import os
from pathlib import Path

audio_dir = Path("/data/akshaj/MusicAI/gaps_v1/audio")
midi_dir = Path("/data/akshaj/MusicAI/gaps_v1/midi")

audio_files = sorted(audio_dir.glob("*.wav"))

print(f"{len(audio_files)} audio files")

matched = 0
unmatched = []

for audio_file in audio_files:
    audio_name = audio_file.stem

    if "_" not in audio_name:
        print(f"Skipping {audio_file.name} - formatting is wrong")
        continue

    parts = audio_name.split("_", 1)
    number = parts[0]
    base_name = parts[1]

    midi_pattern = f"{base_name}-fine-aligned.mid"

    matching_midi = None
    for midi_file in midi_dir.glob("*-fine-aligned.mid"):
        if midi_file.name == midi_pattern:   # exact, case-sensitive
            matching_midi = midi_file
            break


    if matching_midi:
        new_midi_name = f"{number}_{base_name}.mid"
        new_midi_path = midi_dir / new_midi_name

        print(f"Renaming: {matching_midi.name} into {new_midi_name}")
        matching_midi.rename(new_midi_path)
        matched += 1
    else:
        unmatched.append((audio_file.name, base_name))
        print(f"No match found for {audio_file.name}")

print(f"\nMatched and renamed: {matched}")
print(f"Unmatched: {len(unmatched)}")

if unmatched:
    print("\nUnmatched files:")
    for audio_name, base_name in unmatched[:]:
        print(f"{audio_name} (base: {base_name})")