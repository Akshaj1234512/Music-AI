import mido

mid = mido.MidiFile('/data/akshaj/MusicAI/GOAT/midi/item_0__item_0_fine_aligned.mid')
has_bends = False
for track in mid.tracks:
    for msg in track:
        if msg.type == 'pitchwheel':
            print("Found pitch bend data!")
            has_bends = True
            break
if not has_bends:
    print("No pitch bends found. Only Note On/Off.")