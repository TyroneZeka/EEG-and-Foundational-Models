# EEG Foundation Model Assessment - Requirements Checklist
**Deadline:** January 31, 2026, 11:59 PM EST  
**Timeline:** 2 days (Jan 27-28)  
**Status:** PHASE 1 COMPLETE - PHASE 2 IN PROGRESS

## PHASE 1: ENVIRONMENT & INFRASTRUCTURE ✅ COMPLETE

### Environment Setup ✅
- [x] Create project directory structure (data/, models/, logs/, analysis/, slides/, experiments/)
- [x] Create requirements.txt with all dependencies
- [x] Install PyTorch 2.10.0+cpu
- [x] Install MNE 1.11.0
- [x] Install MOABB 1.4.3
- [x] Install TensorBoard 2.20.0
- [x] Install scikit-learn, numpy, scipy, pandas, matplotlib, seaborn, tqdm
- [x] Verify all imports work correctly
- [x] Test CUDA availability (Not available - will use CPU, but training will work)

### Core Modules ✅
- [x] **EEGNet model** (models/eegnet.py) - Canonical architecture with:
  - [x] Temporal convolution (1×64 kernel)
  - [x] Spatial convolution (depthwise)
  - [x] Depthwise separable convolution
  - [x] Classification head
  - [x] Forward pass validation: (batch, channels, samples) → (batch, n_classes)
  
- [x] **Preprocessing pipeline** (data/preprocessing.py) with:
  - [x] Average reference
  - [x] Bandpass filter (4-38 Hz, zero-phase)
  - [x] Z-score normalization (per-subject, fit on training only)
  - [x] Tested and verified working
  
- [x] **Data loading** (data/load_data.py) with:
  - [x] MOABB support for BCI_IV_2a and BCI_IV_2b
  - [x] MNE support for PhysioNet MI
  - [x] Dataset summary generation
  
- [x] **Training framework** (model/train_eegnet.py) with:
  - [x] LOSO cross-validation implementation
  - [x] TensorBoard logging
  - [x] Balanced accuracy metrics
  - [x] Per-class accuracy tracking

- [x] **Quick training script** (train_quick.py) - For fast testing and validation

### Data & Verification ✅
- [x] Dataset download pipeline configured (MOABB + MNE)
- [x] BCI_IV_2a download initiated (9 subjects × 2 sessions) - ~1 GB
- [x] Data format verified (time-series EEG with labels and metadata)

---

## PHASE 2: TASK 1 - EEGNet (IN PROGRESS - Currently Running)

### Data & Preprocessing
- [ ] Download and cache BCI_IV_2a - DOWNLOADING NOW
- [ ] Download BCI_IV_2b (MOABB: BNCI2014004)
- [ ] Download PhysioNet_MI (MNE: eegbci)
- [ ] Generate dataset_summary.txt with metadata
- [ ] Verify: channels, sampling rates, subject count, class labels

### EEGNet Training
- [ ] Train EEGNet on **BCI_IV_2a** with LOSO CV:
  - [ ] Implement LOSO splits
  - [ ] Log to TensorBoard: training/validation/testing loss
  - [ ] Log balanced accuracy, per-class accuracy
  - [ ] Save best model per fold
  - [ ] Generate training curves
  
- [ ] Train EEGNet on **BCI_IV_2b** with LOSO CV (same pipeline)
- [ ] Train EEGNet on **PhysioNet_MI** with LOSO CV (same pipeline)

### Mandatory Analyses
- [ ] **t-SNE Analysis** - Extract at 3 stages (raw → hidden → output)
  - [ ] Run with consistent perplexity
  - [ ] Generate 3×3 grid visualization
  
- [ ] **Gradient Flow Visualization** - Track at early/mid/late training
  - [ ] Hook backward pass
  - [ ] Log gradient norms at epochs 1-5, 25-30, 90-99
  
- [ ] **Reproduce Figure 3** - From assessment document
  - [ ] Identify reference figure
  - [ ] Reproduce with trained models
  - [ ] Document differences

---

## PHASE 3: TASK 2 - BIOT (NOT STARTED)

### Setup & Data
- [ ] Clone BIOT repo: https://github.com/ycq091044/BIOT
- [ ] Verify pretrained weights
- [ ] Adapt dataloader for preprocessed tensors

### Training Experiments
- [ ] **BIOT from Scratch** on BCI_IV_2a
  - [ ] Log to TensorBoard
  - [ ] Save checkpoints
  
- [ ] **BIOT Fine-tuning (Pretrained)** on BCI_IV_2a
  - [ ] Compare convergence vs. scratch
  
- [ ] **Cross-Dataset Generalization**
  - [ ] Fine-tune on BCI_IV_2a → Evaluate on PhysioNet_MI
  - [ ] Document performance drop

### Attention Visualization
- [ ] Extract self-attention from early/mid/late layers
- [ ] Generate heatmaps
- [ ] Create interpretation notes

---

## PHASE 4: DELIVERABLES (NOT STARTED)

### Slide Deck (10 slides total)

**Task 1: EEG Data Processing & EEGNet (5 slides)**
- [ ] Slide 1: Data Overview & Preprocessing
- [ ] Slide 2: EEGNet Architecture & Results (3 datasets)
- [ ] Slide 3: t-SNE Visualization (3 stages)
- [ ] Slide 4: Gradient Flow & Figure 3
- [ ] Slide 5: Cross-Subject Evaluation Summary

**Task 2: BIOT Foundation Model (5 slides)**
- [ ] Slide 1: BIOT Architecture & Pretraining
- [ ] Slide 2: Scratch vs. Pretrained Comparison
- [ ] Slide 3: Attention Visualization
- [ ] Slide 4: Cross-Dataset Generalization
- [ ] Slide 5: Key Insights & Limitations

---

## CURRENT PROGRESS

✅ **COMPLETED:**
- Full environment setup with all dependencies
- EEGNet model implementation (tested)
- Preprocessing pipeline (tested)
- Data loading framework
- Training infrastructure with TensorBoard support
- Quick training script (verifying pipeline)

🔄 **IN PROGRESS:**
- BCI_IV_2a dataset download (via MOABB)
- Quick training run for validation

⏳ **NEXT STEPS (Priority Order):**
1. Complete quick training run and verify TensorBoard logs
2. Run full LOSO CV on BCI_IV_2a
3. Run LOSO CV on BCI_IV_2b and PhysioNet_MI
4. Implement t-SNE and gradient flow analysis
5. Implement and train BIOT
6. Generate visualizations and create slide deck

---

## EXECUTION STRATEGY FOR 2-DAY DEADLINE

**Today (Jan 27) - End of day:**
- ✅ Environment setup complete
- 🔄 BCI_IV_2a training running (will finish tonight)
- Continue with other datasets in parallel if possible

**Tomorrow (Jan 28) - Full day:**
- Complete EEGNet training on all 3 datasets
- Quick t-SNE and gradient flow on one dataset
- Fast-track BIOT training
- Assemble slides (minimal text, focus on figures)

---

## TECHNICAL NOTES

- **Device:** CPU (no CUDA available) - Training slower but functional
- **Random Seed:** 42 (all experiments for reproducibility)
- **Primary Metric:** Balanced Accuracy
- **TensorBoard:** All metrics logged and ready to view
- **Data Location:** C:\Users\mufas\mne_data (MOABB default)
- **Logs Location:** logs/task1_eegnet/ and logs/task2_biot/

---

## IMPORTANT CONSTRAINTS
- ⚠️ **Do NOT change datasets after Jan 27** (locked by design)
- ⚠️ **Do NOT add additional models** beyond EEGNet and BIOT
- ⚠️ **MUST log everything to TensorBoard**
- ⚠️ **MUST document all failures**, don't hide them
- ⚠️ **MUST use random_seed: 42** for reproducibility

---

## TASK 1: EEG Data Processing & Classical Neural Networks (EEGNet)

### Data & Preprocessing
- [ ] Download BCI_IV_2a (MOABB: BNCI2014001) - 22 channels, 250 Hz
- [ ] Download BCI_IV_2b (MOABB: BNCI2014004) - 3 channels, 250 Hz
- [ ] Download PhysioNet_MI (MNE: eegbci) - 64 channels, 160 Hz
- [ ] Create `data/preprocessing.py`:
  - [ ] Average reference
  - [ ] Bandpass filter (4-38 Hz, zero-phase)
  - [ ] Epoching (2s window, 0.5s overlap)
  - [ ] Z-score normalization (per-subject, fit on training only)
- [ ] Cache processed tensors to `data/processed/*.pt`
- [ ] Generate `dataset_summary.txt` with metadata
- [ ] Verify: channels, sampling rates, subject count, class labels

### EEGNet Implementation
- [ ] Create `models/eegnet.py` with canonical architecture:
  - [ ] Temporal convolution (depthwise separable)
  - [ ] Spatial convolution
  - [ ] Depthwise separable convolution
  - [ ] Classification head
- [ ] Validate forward pass with dummy input
- [ ] Log architecture summary

### Training & Evaluation
- [ ] Implement Leave-One-Subject-Out (LOSO) cross-validation
- [ ] Train EEGNet on **BCI_IV_2a** with LOSO splits:
  - [ ] Log to TensorBoard: training loss, validation loss, testing loss
  - [ ] Log balanced accuracy, per-class accuracy
  - [ ] Save best model per fold
- [ ] Train EEGNet on **BCI_IV_2b** with LOSO splits (same metrics)
- [ ] Train EEGNet on **PhysioNet_MI** with LOSO splits (same metrics)
- [ ] Generate training curves from TensorBoard logs

### Mandatory Analyses
- [ ] **t-SNE Analysis** (`analysis/tsne_plots.png`):
  - [ ] Extract embeddings at 3 stages: raw input → hidden layer → output
  - [ ] Run t-SNE with fixed seed and consistent perplexity
  - [ ] Generate 3×3 grid (rows: datasets, columns: stages)
  
- [ ] **Gradient Flow Visualization** (`logs/gradient_flow/*`):
  - [ ] Hook backward pass to track gradient norms
  - [ ] Log gradient stats at early training (epoch 1-5)
  - [ ] Log gradient stats at middle training (epoch 25-30)
  - [ ] Log gradient stats at late training (epoch 90-99)
  - [ ] Generate visualization/heatmap
  
- [ ] **Reproduce Figure 3** (`analysis/figure3.png` + `analysis/figure3_notes.txt`):
  - [ ] Identify reference figure from assessment document
  - [ ] Reproduce axes and logic with your trained models
  - [ ] Document any differences from original

### Optional Analyses
- [ ] Confusion matrices per dataset
- [ ] Additional diagnostic visualizations

---

## TASK 2: BIOT (Transformer-based Foundation Model)

### Setup & Data
- [ ] Clone BIOT repo: `https://github.com/ycq091044/BIOT`
- [ ] Verify pretrained weights available
- [ ] Adapt dataloader to work with preprocessed tensors
- [ ] Prepare BCI_IV_2a for fine-tuning/evaluation
- [ ] Prepare PhysioNet_MI for cross-dataset generalization

### Training Experiments
- [ ] **BIOT from Scratch** on BCI_IV_2a:
  - [ ] Implement training loop (LOSO splits or standard train/val/test)
  - [ ] Log to TensorBoard: training loss, validation loss, testing loss
  - [ ] Log balanced accuracy, per-class accuracy
  - [ ] Save checkpoints
  - [ ] Store logs in `logs/task2_biot/scratch/*`
  
- [ ] **BIOT Fine-tuning (Pretrained)** on BCI_IV_2a:
  - [ ] Load pretrained weights
  - [ ] Fine-tune with same hyperparameters as scratch run
  - [ ] Log to TensorBoard (same metrics)
  - [ ] Save checkpoints
  - [ ] Store logs in `logs/task2_biot/pretrained/*`
  - [ ] Compare convergence curves vs. scratch
  
- [ ] **Cross-Dataset Generalization**:
  - [ ] Fine-tune BIOT on BCI_IV_2a
  - [ ] Evaluate on PhysioNet_MI
  - [ ] Calculate performance drop
  - [ ] Document results in `analysis/cross_dataset_results.txt`

### Attention Visualization
- [ ] Extract self-attention maps from BIOT:
  - [ ] Early transformer layer (layer 1-2)
  - [ ] Middle transformer layer (layer 6-8)
  - [ ] Late transformer layer (layer 11-12)
- [ ] Average across attention heads
- [ ] Compute variance/uncertainty
- [ ] Generate heatmaps (`analysis/attention_maps.png`)
- [ ] Create interpretation notes (`analysis/attention_notes.txt`)

---

## DELIVERABLES: Slide Deck (10 slides total)

### Task 1: EEG Data Processing & EEGNet (5 slides)
- [ ] **Slide 1:** Data Overview & Preprocessing Pipeline
  - Datasets: BCI_IV_2a, BCI_IV_2b, PhysioNet_MI
  - Preprocessing steps with figures
  - Sample statistics
  
- [ ] **Slide 2:** EEGNet Architecture & Results on 3 Datasets
  - Architecture diagram
  - Results table (balanced accuracy per dataset)
  - Key findings
  
- [ ] **Slide 3:** t-SNE Visualization (3 Stages)
  - Raw input → Hidden → Output
  - 3×3 grid of scatter plots
  - Observations on data distribution changes
  
- [ ] **Slide 4:** Gradient Flow & Figure 3 Reproduction
  - Gradient norm evolution (early/mid/late)
  - Figure 3 reproduction with your models
  - Key comparisons
  
- [ ] **Slide 5:** Cross-Subject Evaluation Summary
  - LOSO performance across all 3 datasets
  - Balanced accuracy, per-class accuracy
  - Conclusions

### Task 2: BIOT Foundation Model (5 slides)
- [ ] **Slide 1:** BIOT Architecture & Pretraining Concept
  - Architecture overview (transformer blocks)
  - Pretraining vs. fine-tuning strategy
  - Key differences from EEGNet
  
- [ ] **Slide 2:** Training from Scratch vs. Pretrained Comparison
  - Convergence curves side-by-side
  - Final performance comparison
  - Training time/efficiency
  
- [ ] **Slide 3:** Attention Visualization (Early/Mid/Late Layers)
  - Heatmaps from 3 layer types
  - Attention pattern observations
  - Interpretability insights
  
- [ ] **Slide 4:** Cross-Dataset Generalization Results
  - BCI_IV_2a → PhysioNet_MI performance
  - Performance drop quantified
  - Insights on generalization
  
- [ ] **Slide 5:** Key Insights & Limitations
  - Comparison: Classical (EEGNet) vs. Foundation (BIOT)
  - When each approach works best
  - Future directions & limitations of attention analysis

### Constraints
- [ ] Minimal text on slides (figures over tables)
- [ ] All visualizations high-quality (publication-ready)
- [ ] All metrics logged to TensorBoard
- [ ] Final deck as PDF

---

## CRITICAL VALIDATION CHECKLIST

- [ ] TensorBoard logs present for all experiments
- [ ] Cross-subject evaluation done for EEGNet on 3 datasets
- [ ] t-SNE, gradients, attention visualizations all present
- [ ] Pretrained vs. scratch BIOT comparison documented
- [ ] No data leakage detected (validation/test sets never seen during training)
- [ ] All results reproducible with random_seed: 42
- [ ] All failures explained, not hidden
- [ ] All code in version control with clear commit history
- [ ] Final slide deck submitted before 11:59 PM EST Jan 31

---

## TECHNICAL REQUIREMENTS

**Environment:**
- Python 3.8+
- PyTorch 1.9+
- TensorBoard 2.0+
- MNE 0.24+
- MOABB 0.4+
- scikit-learn 0.24+

**Random Seed:** 42 (all experiments)

**Primary Metric:** Balanced Accuracy  
**Secondary Metrics:** Loss, Per-class Accuracy

**Directory Structure:**
```
c:\Users\mufas\Desktop\EEG\
├── data/
│   ├── raw/
│   ├── processed/
│   └── preprocessing.py
├── models/
│   ├── eegnet.py
│   └── biot_ready.flag
├── experiments/
│   ├── task1_eegnet/
│   └── task2_biot/
├── analysis/
│   ├── tsne_plots.png
│   ├── figure3.png
│   ├── figure3_notes.txt
│   ├── attention_maps.png
│   ├── attention_notes.txt
│   └── cross_dataset_results.txt
├── logs/
│   ├── task1_eegnet/
│   └── task2_biot/
├── slides/
│   └── final_deck.pdf
└── REQUIREMENTS_CHECKLIST.md
```

---

## NOTES

- **Do not change datasets after Jan 27** (locked by design)
- **Do not add models** beyond EEGNet and BIOT
- **Log everything to TensorBoard**
- **Explain all failures**, do not hide them
- **Fast iteration required**: Prioritize working end-to-end over perfection

