import os
import torch
import numpy as np
from model import LSTMModel
from dataset import notes_to_input_sequence, midi_to_note_sequence
from mido import Message, MidiFile, MidiTrack


input_size = 128
hidden_size = 256
output_size = 128
sequence_length = 100
midi_file_path = 'LofiData/1.mid'

model = LSTMModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('lstm_model.pth'))
model.eval()

def generate_music(model, start_sequence, length):
    generated = start_sequence.copy()
    input_seq = torch.tensor(start_sequence, dtype=torch.float32).unsqueeze(0)
    
    for _ in range(length):
        with torch.no_grad():
            output = model(input_seq)
        new_note = output.squeeze().numpy()
        generated = np.vstack([generated, new_note])
        input_seq = torch.tensor(generated[-sequence_length:], dtype=torch.float32).unsqueeze(0)
    
    return generated

def sequence_to_midi(sequence):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    for note in sequence:
        note_number = np.argmax(note)
        track.append(Message('note_on', note=note_number, velocity=64, time=480))
        track.append(Message('note_off', note=note_number, velocity=64, time=480))
    
    return mid

initial_notes = midi_to_note_sequence(midi_file_path)
start_sequence = notes_to_input_sequence(initial_notes)[:sequence_length]


generated_music = generate_music(model, start_sequence, 100)
generated_midi = sequence_to_midi(generated_music)
generated_midi.save('generated_music.mid')

print("Music generation complete. The generated music has been saved as 'generated_music.mid'.")
