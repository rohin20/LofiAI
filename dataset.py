import mido
import numpy as np
import os

def midi_to_note_sequence(file_path):
    midi = mido.MidiFile(file_path)
    notes = []
    for track in midi.tracks:
        for msg in track:
            if msg.type in ['note_on', 'note_off']:
                notes.append(msg)
    return notes

def notes_to_input_sequence(notes):
    sequence = []
    for note in notes:
        one_hot = np.zeros(128)  
        one_hot[note.note] = 1
        sequence.append(one_hot)
    return np.array(sequence)

def load_midi_files(directory):
    sequences = []
    for filename in os.listdir(directory):
        if filename.endswith('.mid'):
            notes = midi_to_note_sequence(os.path.join(directory, filename))
            sequences.append(notes_to_input_sequence(notes))
    return sequences

def create_training_data(sequences, sequence_length):
    data = []
    targets = []
    for notes_sequence in sequences:
        for i in range(len(notes_sequence) - sequence_length):
            data.append(notes_sequence[i:i+sequence_length])
            targets.append(notes_sequence[i+sequence_length])
    return np.array(data), np.array(targets)