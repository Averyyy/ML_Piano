import numpy as np
import mido

def load_midi(filepath):
    mid = mido.MidiFile(filepath)
    notes = []
    for msg in mid.play():
        if msg.type == 'note_on':
            notes.append((msg.note, msg.velocity))
    return np.array(notes)

def save_midi(notes, filepath):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    for note, velocity in notes:
        track.append(mido.Message('note_on', note=note, velocity=velocity))
    mid.save(filepath)