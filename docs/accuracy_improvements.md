# ASL Translator Accuracy Improvements

## What Was Wrong

### Problem 1: No Landmark Normalization
**Issue:** Raw MediaPipe coordinates (0-1 range based on frame position) were used directly.

**Impact:** The same sign at different positions in the frame produced different feature values. Moving your hand left, right, closer, or further caused misclassifications.

**Fix:** Implemented wrist-centered, palm-scaled normalization:
- Subtract wrist position from all landmarks (centers at origin)
- Divide by palm width (distance from index MCP to pinky MCP)
- Features now describe hand *shape* not *position*

### Problem 2: Frame Mirroring During Prediction
**Issue:** The webcam frame was flipped horizontally before detection to create a mirror-like experience.

**Impact:** A right hand appeared as a left hand to the model. Signs that differ by orientation (L vs Q, etc.) were systematically confused.

**Fix:** 
- Detection uses the original (non-flipped) frame
- Display uses the flipped frame for natural viewing
- Skeleton is correctly drawn on the flipped display

### Problem 3: Domain Mismatch
**Issue:** Model trained on Kaggle images (studio lighting, white backgrounds, centered hands) but ran on webcam data (variable lighting, backgrounds, positions).

**Impact:** Features that worked in training didn't generalize to real-world conditions.

**Fix:**
- Normalization reduces position/scale variance
- Personal data collection captures YOUR hand, YOUR camera, YOUR lighting
- Mixed training combines Kaggle variety with personal specificity

### Problem 4: Prediction Flickering
**Issue:** Single-frame predictions are noisy. Minor landmark jitter caused rapid prediction changes.

**Impact:** Even holding a steady sign, the displayed letter would flicker between options.

**Fix:** Prediction smoothing using majority vote:
- Maintains a sliding window of last 7 predictions
- Only displays a prediction if it has 50%+ agreement
- Averages confidence across agreeing frames
- Clears history when hand leaves frame

---

## Updated Pipelines

### Training Pipeline

```
1. Kaggle images → MediaPipe landmarks → Normalize → data/processed_normalized/
2. Personal webcam → MediaPipe landmarks → Normalize → data/personal/
3. Combine (personal 2x weighted) → Train RF (300 trees, balanced classes)
4. Save → models/saved/classifier.joblib
```

**Files:**
- `scripts/normalize_dataset.py` - Normalize Kaggle data
- `scripts/collect_personal_data.py` - Collect your own data
- `scripts/train_mixed.py` - Train on combined data

### Inference Pipeline

```
1. Webcam frame (original, NOT flipped)
2. MediaPipe hand detection
3. Extract 63 landmarks (21 × 3)
4. Normalize (wrist-center, palm-scale)
5. Random Forest prediction + confidence
6. Prediction smoother (7-frame window, 50% agreement)
7. Display on flipped frame with skeleton
```

**Files:**
- `backend/inference/asl_translator.py` - Main translator with all fixes

---

## How to Use

### Step 1: Collect Personal Data
```bash
cd /path/to/project
source venv/bin/activate
python scripts/collect_personal_data.py
```
- Signs each letter A-Z, 0-9
- 150 samples per sign (takes ~30 min)
- Saved to `data/personal/`

### Step 2: Retrain with Mixed Data
```bash
python scripts/train_mixed.py
```
- Combines Kaggle + personal data
- Weights personal data 2x
- Saves new model to `models/saved/`

### Step 3: Run Improved Translator
```bash
cd backend/inference
python asl_translator.py
```

---

## Expected Accuracy Improvements

| Stage | Test Accuracy | Notes |
|-------|--------------|-------|
| Original (raw coords) | ~92% | High on Kaggle, poor real-time |
| + Normalization | ~86% | Lower test, better generalization |
| + Mirror fix | ~86% | Fixes orientation confusions |
| + Personal data | ~95%+ | Tuned to YOUR hand |
| + Smoothing | — | Reduces visual noise |

---

## Files Changed

### New Files Created:
- `backend/preprocessing/normalize.py` - Normalization functions
- `backend/diagnostics/pipeline_check.py` - Pipeline comparison tool
- `backend/diagnostics/confusion_analyzer.py` - Live confusion analysis
- `scripts/normalize_dataset.py` - Normalize training data
- `scripts/collect_personal_data.py` - Personal data collection
- `scripts/train_mixed.py` - Mixed data training
- `docs/domain_mismatch_analysis.md` - Domain mismatch explanation
- `docs/accuracy_improvements.md` - This file

### Modified Files:
- `backend/inference/asl_translator.py`:
  - Added `normalize_landmarks()` function
  - Added `PredictionSmoother` class
  - Fixed frame mirroring (detect on original, display on flipped)
  - Integrated smoother into main loop

---

## Key Takeaways

1. **Normalize landmarks** - Position/scale invariance is critical
2. **Watch for mirroring** - Display mirroring shouldn't affect prediction
3. **Collect domain-specific data** - Best results come from training on similar data
4. **Smooth predictions** - Temporal filtering reduces noise
5. **Separate detection from display** - Keep processing pipeline clean
