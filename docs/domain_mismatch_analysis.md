# Domain Mismatch Analysis & Solutions

## Problem: Training vs Real-World Gap

Your model was trained on Kaggle images but runs on live webcam data. These differ in:

### 1. Image Characteristics

| Factor | Kaggle Dataset | Your Webcam |
|--------|---------------|-------------|
| Background | White/uniform | Variable (room, objects) |
| Lighting | Studio lighting | Natural/artificial, variable |
| Hand position | Centered, cropped | Anywhere in frame |
| Hand size | Fills frame | Variable distance |
| Hand orientation | Consistent | Variable |
| Image quality | High, static | Motion blur possible |

### 2. Hand Differences

| Factor | Kaggle Dataset | Your Hand |
|--------|---------------|-----------|
| Skin tone | Varied (dataset dependent) | Your specific skin tone |
| Hand size | Various | Your hand size |
| Finger proportions | Various | Your proportions |
| Jewelry/accessories | Likely none | Possible rings, watch |

### 3. Camera Differences

| Factor | Kaggle Dataset | Your Setup |
|--------|---------------|------------|
| Resolution | Various | Your webcam resolution |
| Lens distortion | Various | Your webcam lens |
| Color balance | Various | Your camera's auto-WB |
| Mirroring | Likely not mirrored | Mirrored for natural viewing |

---

## Solutions Implemented

### ✅ Solution 1: Landmark Normalization
**Status: IMPLEMENTED**

Instead of using raw (x, y, z) coordinates which vary based on hand position and size:
- Center landmarks around wrist (landmark 0)
- Scale by palm width (distance between index MCP and pinky MCP)

This makes predictions invariant to:
- Hand position in frame
- Distance from camera
- Hand size differences

### ✅ Solution 2: Mirror Handling  
**Status: PARTIALLY IMPLEMENTED**

The webcam feed is flipped horizontally so gestures feel natural. However:
- If training data shows right hands and you're using your right hand, the flip makes it appear as a left hand
- This can cause L/Q, and other orientation-dependent signs to be confused

**Fix:** Either:
1. Don't flip the frame for prediction (only for display)
2. Or train with mirrored data augmentation

---

## Additional Solutions to Consider

### Solution 3: Data Augmentation (Future)
Augment training data with:
- Brightness variations
- Contrast variations  
- Slight rotations
- Different hand positions (even though normalized)

### Solution 4: Collect Your Own Data (Recommended)
Best accuracy comes from training on data similar to deployment:
1. Use the webcam capture tool
2. Sign each letter/number 20-50 times
3. Vary hand position, distance, angle
4. Combine with Kaggle data or use alone

### Solution 5: Temporal Smoothing (Future)
Single-frame predictions are noisy. Implement:
- Rolling average over last N predictions
- Require same prediction for K consecutive frames
- Weight by confidence

---

## How to Test Improvements

1. Run the confusion analyzer:
```bash
cd backend/diagnostics && python confusion_analyzer.py
```

2. Sign each letter multiple times
3. Record actual vs predicted
4. Check which letters are commonly confused
5. Analyze patterns (similar hand shapes? orientation issues?)

---

## Expected Confusion Pairs

Based on ASL hand shapes, these are commonly confused:
- **M, N, S, T, A, E** - Closed fist variants
- **U, V, R** - Two finger pointing variants
- **G, Q** - Similar shape, different orientation
- **D, 1** - Very similar
- **W, 6** - Similar finger spread

If you see L→Q confusion, it's likely a mirror/orientation issue.
