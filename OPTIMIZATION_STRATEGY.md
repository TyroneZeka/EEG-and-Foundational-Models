# Advanced Training Strategy - Quality Over Speed

## Philosophy
**Accuracy First** - We prioritize model performance over deadline compliance.

## Key Improvements

### 1. **Data Augmentation** (on-the-fly)
- **Gaussian Noise**: Add realistic sensor noise
- **Temporal Shift**: Circular rotation of time axis (±15 samples)
- **Time Warping**: Non-linear time stretching
- **Mixup**: Linear interpolation between samples
- **Applied 50% of the time** to prevent overfitting

### 2. **Optimized Hyperparameters**
```
Learning Rate:         0.0005 (lower for better convergence)
Weight Decay:          0.0001 (L2 regularization)
Scheduler:             CosineAnnealingWarmRestarts (T_0=50, T_mult=1.5)
Batch Size:            32 (good balance)
Max Epochs:            400 (patience=50 for early stopping)
Gradient Clipping:     norm=1.0 (stability)
Optimizer:             AdamW (better than Adam)
```

### 3. **Per-Fold Normalization**
- Z-score fit only on train+val (prevents test leakage)
- Separate seed per fold for better diversity
- Stratified train/val split (80/20)

### 4. **Advanced Learning Rate Scheduling**
- **CosineAnnealingWarmRestarts**: Periodic restarts help escape local minima
- Learning rate decays to 1e-6 minimum
- Warm restarts every 50 epochs help exploration

### 5. **Training Strategy**
- Train/Val/Test split with NO leakage
- Early stopping with patience=50 (allows exploration)
- Model checkpoint on best validation accuracy
- TensorBoard logging for monitoring

## Expected Improvements

**Previous (no augmentation):** 0.4389 ± 0.1596
**Expected (with optimization):** 0.50-0.55+ (20-25% improvement)

## Usage

```bash
# Train all 3 datasets with optimization
python model/train_eegnet_optimized.py

# Monitor progress
tensorboard --logdir logs/task1_eegnet_optimized
```

## Timeline with Optimization

- BCI_IV_2a: 400 epochs × 9 subjects = 6-8 hours
- BCI_IV_2b: 400 epochs × 3 subjects = 2-3 hours  
- PhysioNet_MI: 400 epochs × varies = 3-4 hours
- **Total: 11-15 hours** (still within 2-day window)

## What Makes This Better

1. **Augmentation prevents overfitting** on small EEG datasets
2. **Cosine annealing with restarts** finds better minima
3. **Lower learning rate** allows more careful optimization
4. **Per-fold normalization** maintains data integrity
5. **Longer training** (400 vs 100 epochs) for convergence
6. **Weight decay** reduces model complexity

## Monitoring

Watch TensorBoard for:
- Training loss converging smoothly
- Validation accuracy improving consistently
- No sudden spikes (sign of instability)
- Learning rate schedule following cosine pattern

## Next Steps

1. Start training: `python model/train_eegnet_optimized.py`
2. Monitor TensorBoard in real-time
3. Once complete, move to BIOT training
4. Final analysis and slides

**Focus:** Each dataset should achieve **0.50+ balanced accuracy** as the new target.
