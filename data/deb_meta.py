import mne
import pandas as pd
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)

mne.set_log_level("ERROR")

def inspect_metadata():
    """
    Loads data and prints the contents of the .metadata DataFrame to find the correct column names.
    """
    dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[1])
    
    preprocess(dataset, [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        Preprocessor(fn=lambda x: x * 1e6),
        Preprocessor("resample", sfreq=250),
    ])
    
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=1000,
        window_stride_samples=1000,
        preload=True,
    )
    
    subject_dataset = windows_dataset.datasets[0]

    # --- DIAGNOSTIC ---
    print("--- Metadata DataFrame Inspection ---")
    metadata_df = subject_dataset.metadata
    
    print("\n[INFO] First 5 rows of the .metadata DataFrame:")
    print(metadata_df.head())
    
    print("\n[INFO] Actual column names in .metadata DataFrame:")
    print(metadata_df.columns.tolist())


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message="Preprocessing choices with lambda functions cannot be saved.")
    inspect_metadata()
