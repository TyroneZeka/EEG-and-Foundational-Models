import mne
import pandas as pd
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)

# Suppress verbose MNE logging for cleaner output
mne.set_log_level("ERROR")

def load_and_preprocess_data():
    """
    Downloads, preprocesses, and windows the BNCI2014-001 dataset for a single subject.
    """
    # 1. Load the raw data
    dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[1])

    # 2. Preprocess the raw data
    raw_preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        Preprocessor(fn=lambda x: x * 1e6), # Convert from V to µV
        Preprocessor("resample", sfreq=250),
    ]
    preprocess(dataset, raw_preprocessors)

    # 3. Create 4-second trials (windows)
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=1000,
        window_stride_samples=1000,
        preload=True,
    )

    # 4. === THIS IS THE CORRECTED SECTION ===
    # Get the inner dataset for our single subject
    subject_dataset = windows_dataset.datasets[0]

    # Get shape from the first trial's data array. An item is a (data, label, metadata) tuple.
    single_trial_data = subject_dataset[0][0]
    window_shape_tuple = single_trial_data.shape  # (channels, time_samples)

    # The sampling frequency is stored in the metadata DataFrame called 'description'
    sfreq = subject_dataset.description['sfreq'].iloc[0]

    # Reconstruct the class mapping from the metadata
    labels_df = subject_dataset.description[['event_name', 'target']].drop_duplicates()
    # Sort by target to ensure consistent order
    labels_df = labels_df.sort_values(by='target')
    class_mapping = {row['event_name']: row['target'] for _, row in labels_df.iterrows()}


    print("--- Dataset Information ---")
    print(f"Number of windows (trials): {len(subject_dataset)}")
    print(f"Window shape (Channels, Time samples): {window_shape_tuple}")
    print(f"Sampling frequency: {sfreq} Hz")
    print(f"Classes (labels): {class_mapping}")


if __name__ == "__main__":
    # The UserWarning about lambda functions is expected and can be ignored.
    # We add this filter to hide it for a cleaner output.
    import warnings
    warnings.filterwarnings("ignore", message="Preprocessing choices with lambda functions cannot be saved.")
    load_and_preprocess_data()
