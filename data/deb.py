import mne
import pandas as pd
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)

mne.set_log_level("ERROR")

def inspect_final_object():
    """
    Inspects the top-level 'windows_dataset' object to find the class mapping.
    """
    dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[1])
    
    preprocess(dataset, [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        Preprocessor(fn=lambda x: x * 1e6),
        Preprocessor("resample", sfreq=250),
    ])
    
    # This function creates the object that holds the mapping.
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=1000,
        window_stride_samples=1000,
        preload=True,
    )
    
    # --- DIAGNOSTIC ---
    print("--- Top-Level 'windows_dataset' Inspection ---")
    
    # This is the DataFrame that should contain all necessary info.
    description_df = windows_dataset.description
    
    print("\n[INFO] First 5 rows of the top-level description DataFrame:")
    print(description_df.head())
    
    print("\n[INFO] Actual column names in the top-level description DataFrame:")
    print(description_df.columns.tolist())

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message="Preprocessing choices with lambda functions cannot be saved.")
    inspect_final_object()
