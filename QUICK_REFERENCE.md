# Quick Reference Guide - EEG Assessment Commands

## Essential Commands

### 1️⃣ Monitor Training Progress
```bash
# View TensorBoard (open in browser: http://localhost:6006)
tensorboard --logdir logs/task1_eegnet/quick_test

# Once full training starts:
tensorboard --logdir logs/task1_eegnet
```

### 2️⃣ Start Full LOSO CV Training
```bash
# After quick training completes
python model/train_eegnet.py
```

### 3️⃣ Check System Status
```bash
# Verify all imports work
python test_setup.py

# Check what's downloaded
python -c "import os; print(os.listdir(r'C:\Users\mufas\mne_data\MNE-bnci-data\database\data-sets\001-2014'))"
```

### 4️⃣ Dataset Verification
```bash
# After data loads, check metadata
python -c "from data.load_data import DataLoader; l = DataLoader(); d = l.load_bci_2a(); print(d.keys()); print(d['X'].shape, d['y'].shape)"
```

---

## File Locations

### Data
- **Downloaded:** `C:\Users\mufas\mne_data\MNE-bnci-data\`
- **Cached:** `data/processed/` (will create when preprocessing)

### Logs (TensorBoard)
- **Quick Test:** `logs/task1_eegnet/quick_test/`
- **Full Training:** `logs/task1_eegnet/{dataset}/fold_{N}/`
- **BIOT:** `logs/task2_biot/`

### Models
- **Implementation:** `models/eegnet.py`
- **Saved Models:** `experiments/task1_eegnet/` and `experiments/task2_biot/`

### Documentation
- **Main Checklist:** `REQUIREMENTS_CHECKLIST.md`
- **Phase 1 Summary:** `PHASE_1_SUMMARY.md`
- **Status Report:** `STATUS_REPORT.md`
- **This Guide:** `QUICK_REFERENCE.md`

---

## Key Python Classes

### EEGNet
```python
from models.eegnet import EEGNet
import torch

# Create model
model = EEGNet(
    n_channels=22,    # EEG channels
    n_classes=4,      # Motor imagery classes
    n_samples=1000    # Samples per trial
)

# Forward pass
x = torch.randn(batch_size, 22, 1000)
logits = model(x)  # Shape: (batch_size, 4)

# Get intermediate features for t-SNE
features = model.get_intermediate_features(x, layer='block2_out')
```

### Preprocessing
```python
from data.preprocessing import EEGPreprocessor
import numpy as np

# Create preprocessor
preprocessor = EEGPreprocessor(sampling_rate=250)

# Preprocess data
X, y = preprocessor.preprocess(X_raw, y_labels)

# Get statistics
stats = preprocessor.get_temporal_statistics(X)
```

### Data Loading
```python
from data.load_data import DataLoader

# Load datasets
loader = DataLoader()
datasets = loader.load_all()

# Or load specific dataset
data_2a = loader.load_bci_2a()
# Returns: {'X': array, 'y': labels, 'metadata': dict, ...}
```

### Training
```python
from model.train_eegnet import EEGNetTrainer, loso_cross_validation

# Run LOSO CV
fold_results = loso_cross_validation(
    X, y, metadata,
    dataset_name='BCI_IV_2a',
    log_dir='logs/task1_eegnet'
)

# Access results
for result in fold_results:
    print(f"Fold {result['fold']}: Test accuracy = {result['test_balanced_acc']:.4f}")
```

---

## Common Issues & Solutions

### Issue: Data Takes Too Long to Download
**Solution:** This is normal. First-time download: ~15-30 minutes for BCI_IV_2a  
**Workaround:** Let it run overnight, check progress with `tensorboard --logdir logs`

### Issue: Out of Memory Error
**Solution:** Not expected for this task (CPU has plenty of RAM)  
**Check:** Run `test_setup.py` to verify system is OK

### Issue: TensorBoard Won't Start
**Solution:** 
```bash
# Make sure logs directory exists
python -c "import os; os.makedirs('logs/task1_eegnet', exist_ok=True)"

# Check port 6006 is free
netstat -ano | findstr :6006  # Windows
# If occupied, use different port:
tensorboard --logdir logs/task1_eegnet --port 6007
```

### Issue: Training Seems Stuck
**Solution:**
```bash
# Check if process is running
tasklist | findstr python  # Windows

# Check TensorBoard for loss curves
tensorboard --logdir logs/task1_eegnet/quick_test

# If truly stuck, interrupt (Ctrl+C) and restart
```

---

## Checklist for Next Steps

When quick training completes:

- [ ] Check TensorBoard logs (look for training curves)
- [ ] Verify test accuracy > 60% (baseline)
- [ ] Note any issues or warnings
- [ ] Update REQUIREMENTS_CHECKLIST.md with status
- [ ] Commit changes to git (if using version control)
- [ ] Start full BCI_IV_2a LOSO CV training
- [ ] Let run overnight (4-6 hours expected)

---

## Useful Debugging Commands

```bash
# List what's currently running
ps aux | grep python  # Linux/Mac
tasklist | findstr python  # Windows

# Check disk space
df -h /path/to/mne_data  # Linux/Mac
dir C:\Users\mufas\mne_data  # Windows

# Check memory usage
free -h  # Linux/Mac
Get-ComputerInfo | Select-Object OSTotalVisibleMemorySize, OSFreePhysicalMemory  # Windows

# Clear cached data (WARNING: will require re-download)
rm -rf C:\Users\mufas\mne_data  # Only if needed

# Python: Check installed packages
python -m pip list | grep torch
python -m pip list | grep mne
python -m pip list | grep moabb
```

---

## Configuration Files

### Random Seed (Reproducibility)
- Set in: `model/train_eegnet.py` line ~72
- Value: 42 (fixed)
- ⚠️ **Do not change**

### Learning Rate
- Set in: `model/train_eegnet.py` line ~200
- Value: 0.001
- ⚠️ Only change if models not converging

### Batch Size
- Set in: `model/train_eegnet.py` line ~185
- Value: 32
- ⚠️ Only change if memory issues

### Number of Epochs
- Set in: `train_quick.py` line ~290
- Value: 30 for quick test, 100 for full training
- ✅ Can adjust as needed

---

## Expected Runtimes

| Task | Time | Notes |
|------|------|-------|
| Setup | 30 min | One-time |
| Download BCI_IV_2a | 15-30 min | First-time only |
| Quick Training | 1-2 hours | Validation run |
| BCI_IV_2a LOSO CV | 4-6 hours | 9 folds × 30 epochs |
| BCI_IV_2b LOSO CV | 2-3 hours | 3 folds × 30 epochs |
| PhysioNet MI LOSO CV | 3-4 hours | ~25 folds × 30 epochs |
| t-SNE Analysis | 1 hour | Per dataset |
| BIOT Setup | 30 min | Clone + adapt |
| BIOT Training | 3-4 hours | Scratch + pretrained |
| Slide Assembly | 1-2 hours | Final step |

**Total:** ~25-35 hours  
**Available:** 48 hours (Jan 27 - Jan 31)  
**Margin:** Comfortable ✅

---

## Communication Checkpoints

Report status at:
- [ ] After quick training completes (expected: ~20:00 today)
- [ ] After BCI_IV_2a LOSO CV (expected: ~04:00 tomorrow)
- [ ] After all datasets trained (expected: ~12:00 tomorrow)
- [ ] After BIOT training (expected: ~18:00 tomorrow)
- [ ] After visualization (expected: ~22:00 tomorrow)
- [ ] Final submission (before Jan 31 23:59)

---

## Emergency Contacts/Resources

If stuck:
1. Check PHASE_1_SUMMARY.md for architectural decisions
2. Check STATUS_REPORT.md for current state
3. Check REQUIREMENTS_CHECKLIST.md for what's left
4. Check logs in TensorBoard for detailed metrics
5. Review error messages in terminal output

---

*Quick Reference Generated: January 27, 2026*  
*Last Updated: Phase 1 Complete*
