# predict.py
from keras.models import load_model
import numpy as np
from midiutil import MIDIFile
import pickle
import time

class MIDIGenerator:
    def __init__(self, model_path, mapping_path):
        self.model = load_model(model_path + '.keras')
        with open(mapping_path, 'rb') as f:
            data = pickle.load(f)
            self.note_mapping = data['note_mapping']
            self.reverse_mapping = data['reverse_mapping']
            self.sequence_length = data['sequence_length']
            self.bpm = data['bpm']
            self.beats_per_bar = data['beats_per_bar']
            self.total_bars = data['total_bars']
        
        self.seconds_per_beat = 60 / self.bpm
        self.quarter_note = self.seconds_per_beat
        self.eighth_note = self.quarter_note / 2
    
    def _sample_with_temperature(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds + 1e-7) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        return np.random.choice(len(preds), p=preds)
    
    def generate(self, temperature=0.8):
        # Initialize with a sequence from the valid note range
        current_sequence = []
        current_beat = 0
        
        # Initialize first sequence with quarter notes
        for _ in range(self.sequence_length):
            note = {
                'pitch': np.random.choice(list(self.note_mapping.keys())),
                'duration_beats': 1.0,  # quarter note
                'velocity': 80,
                'time_since_previous': 1.0,  # quarter note spacing
                'start_beat': current_beat
            }
            current_sequence.append(note)
            current_beat += 1
        
        generated_notes = []
        current_beat = 0
        
        # Generate exactly 4 bars worth of notes
        while current_beat < (self.total_bars * self.beats_per_bar):
            sequence_features = []
            for note in current_sequence:
                features = [
                    self.note_mapping[note['pitch']] / len(self.note_mapping),
                    note['duration_beats'] / self.beats_per_bar,
                    note['velocity'] / 127.0,
                    note['time_since_previous'] / self.beats_per_bar
                ]
                sequence_features.append(features)
            
            sequence = np.array([sequence_features])
            pred_probs = self.model.predict(sequence, verbose=0)[0]
            pred_idx = self._sample_with_temperature(pred_probs, temperature)
            
            # Determine if we should use quarter or eighth note based on beat position
            is_strong_beat = current_beat % 1.0 == 0
            duration = 1.0 if is_strong_beat else 0.5  # quarter or eighth note
            
            new_note = {
                'pitch': self.reverse_mapping[pred_idx],
                'duration_beats': duration,
                'velocity': 80,
                'time_since_previous': duration,
                'start_beat': current_beat
            }
            
            generated_notes.append(new_note)
            current_beat += duration
            
            # Update sequence
            current_sequence = current_sequence[1:] + [new_note]
        
        return self._create_midi(generated_notes)
    
    def _create_midi(self, notes):
        midi = MIDIFile(1)
        track = 0
        time = 0
        
        midi.addTempo(track, time, self.bpm)
        
        for note in notes:
            # Convert beat times to seconds for MIDI file
            start_time = note['start_beat'] * self.seconds_per_beat
            duration = note['duration_beats'] * self.seconds_per_beat
            
            midi.addNote(
                track=0,
                channel=0,
                pitch=note['pitch'],
                time=start_time,
                duration=duration,
                volume=note['velocity']
            )
        
        return midi
    
    def save_midi(self, midi_data, output_path):
        with open(output_path, 'wb') as f:
            midi_data.writeFile(f)

def generate_midi(model_path, mapping_path, output_path, temperature=0.8):
    """Generate exactly 4 bars of MIDI music"""
    generator = MIDIGenerator(model_path, mapping_path)
    midi_data = generator.generate(temperature=temperature)
    generator.save_midi(midi_data, output_path)

if __name__ == "__main__":
    for _ in range(20):
        current_time = int(time.time())
        output_path = f"generated/basic-main-model-expirement-{current_time}.mid"
        generate_midi(
            model_path="model",
            mapping_path="mapping.pkl",
            output_path=output_path,
            temperature=0.8
        )