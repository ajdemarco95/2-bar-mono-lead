import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Input
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import pickle
import glob
import pretty_midi
import os
from sklearn.model_selection import train_test_split

class MIDIModelTrainer:
    def __init__(self):
        self.model = None
        self.note_mapping = {}
        self.reverse_mapping = {}
        self.bpm = 120
        self.beats_per_bar = 4
        self.total_bars = 4
        self.seconds_per_beat = 60 / self.bpm
        self.sequence_length = None  # Will be determined from data
        
    def extract_notes(self, midi_path):
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            notes = []
            
            for instrument in midi_data.instruments:
                if not instrument.is_drum:
                    notes_list = []
                    for note in instrument.notes:
                        # Convert time to beats for easier musical analysis
                        start_beat = note.start * (self.bpm / 60)
                        duration_beats = (note.end - note.start) * (self.bpm / 60)
                        
                        notes_list.append({
                            'pitch': note.pitch,
                            'start_beat': start_beat,
                            'duration_beats': duration_beats,
                            'velocity': note.velocity,
                            'bar': int(start_beat / self.beats_per_bar)
                        })
                    
                    # Only keep notes within the 4-bar range
                    notes_list = [n for n in notes_list if n['bar'] < self.total_bars]
                    
                    # Sort notes by start time
                    notes_list = sorted(notes_list, key=lambda x: x['start_beat'])
                    
                    # Calculate time since previous note in beats
                    for i in range(1, len(notes_list)):
                        notes_list[i]['time_since_previous'] = (
                            notes_list[i]['start_beat'] - notes_list[i-1]['start_beat']
                        )
                    if notes_list:
                        notes_list[0]['time_since_previous'] = 0
                    
                    notes.extend(notes_list)
            
            return pd.DataFrame(notes).sort_values('start_beat')
        except Exception as e:
            print(f"Error processing {midi_path}: {str(e)}")
            return None
    
    def prepare_sequences(self, notes_df):
        # Create mapping for notes
        unique_pitches = sorted(notes_df['pitch'].unique())
        self.note_mapping = {note: idx for idx, note in enumerate(unique_pitches)}
        self.reverse_mapping = {idx: note for note, idx in self.note_mapping.items()}
        
        # Set sequence length to 4 beats (one bar) worth of sixteenth notes
        # In 4/4 time, this means 16 positions per sequence
        self.sequence_length = 16  # 4 beats * 4 sixteenth notes per beat
        print(f"Using sequence length of {self.sequence_length} (one bar of sixteenth notes)")
        
        sequences = []
        next_notes = []
        
        # Group notes into sequences within the 4-bar limit
        max_start_beat = self.total_bars * self.beats_per_bar
        filtered_df = notes_df[notes_df['start_beat'] < max_start_beat]
        
        # Quantize notes to sixteenth note grid
        filtered_df['quantized_start'] = (filtered_df['start_beat'] * 4).round() / 4
        
        for i in range(len(filtered_df) - self.sequence_length):
            sequence = filtered_df.iloc[i:i + self.sequence_length]
            next_note = filtered_df.iloc[i + self.sequence_length]
            
            # Only use sequences that stay within the 4-bar limit
            if next_note['start_beat'] < max_start_beat:
                seq_features = []
                for _, note in sequence.iterrows():
                    features = [
                        self.note_mapping[note['pitch']] / len(self.note_mapping),
                        note['duration_beats'] / self.beats_per_bar,  # Normalize to bar length
                        note['velocity'] / 127.0,
                        min(note['time_since_previous'], self.beats_per_bar) / self.beats_per_bar
                    ]
                    seq_features.append(features)
                
                sequences.append(seq_features)
                next_notes.append(self.note_mapping[next_note['pitch']])
        
        X = np.array(sequences)
        y = to_categorical(next_notes, num_classes=len(self.note_mapping))
        
        return X, y
    def build_model(self, input_shape, num_classes):
        inputs = Input(shape=(input_shape[1], input_shape[2]))
        
        x = LSTM(256, return_sequences=True)(inputs)
        x = Dropout(0.3)(x)
        x = LSTM(256, return_sequences=True)(x)
        x = Dropout(0.3)(x)
        x = LSTM(128)(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        return self.model
    
    def train(self, midi_folder_path, epochs=100, batch_size=32):
        # Ensure the MIDI folder exists
        if not os.path.exists(midi_folder_path):
            raise ValueError(f"MIDI folder '{midi_folder_path}' not found")
            
        print(f"Loading MIDI files from {midi_folder_path}")
        midi_files = glob.glob(f"{midi_folder_path}/*.mid")
        if not midi_files:
            raise ValueError(f"No MIDI files found in {midi_folder_path}")
            
        print(f"Found {len(midi_files)} MIDI files")
        all_notes = []
        
        for midi_file in midi_files:
            print(f"Processing {os.path.basename(midi_file)}")
            notes_df = self.extract_notes(midi_file)
            if notes_df is not None:
                all_notes.append(notes_df)
        
        if not all_notes:
            raise ValueError("No valid MIDI files could be processed")
        
        print("Preparing training sequences...")
        combined_notes = pd.concat(all_notes, ignore_index=True)
        X, y = self.prepare_sequences(combined_notes)
        
        print(f"Total sequences: {len(X)}")
        print(f"Features per note: {X.shape[2]}")
        print(f"Unique pitches: {len(self.note_mapping)}")
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        
        print("Building and training model...")
        self.build_model((X.shape[0], X.shape[1], X.shape[2]), y.shape[1])
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
            verbose=1
        )
        
        return history
    
    def save_model(self, model_path, mapping_path):
        print(f"Saving model to {model_path}.keras")
        self.model.save(model_path + '.keras')
        
        print(f"Saving mappings to {mapping_path}")
        with open(mapping_path, 'wb') as f:
            pickle.dump({
                'note_mapping': self.note_mapping,
                'reverse_mapping': self.reverse_mapping,
                'sequence_length': self.sequence_length,
                'bpm': self.bpm,
                'beats_per_bar': self.beats_per_bar,
                'total_bars': self.total_bars
            }, f)

if __name__ == "__main__":
    print("Starting MIDI model training...")
    print("Configuration:")
    print("- BPM: 145")
    print("- Time Signature: 4/4")
    print("- Bars per sequence: 4")
    
    # Create an instance of the trainer
    trainer = MIDIModelTrainer()
    
    try:
        # Train the model
        print("\nInitiating training process...")
        history = trainer.train("midi")
        
        # Save the trained model and mappings
        print("\nSaving model and mappings...")
        trainer.save_model("model", "mapping.pkl")
        
        print("\nTraining completed successfully!")
        print("You can now use predict.py to generate new MIDI sequences.")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        exit(1)