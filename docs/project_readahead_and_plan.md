# CS4100 Project Read-Ahead and Implementation Plan

## Project Context
Project: computer vision-powered hand gesture video playback control, expanded to include motion-based gestures and Raspberry Pi inference.

Primary goal: build an interpretable, modular, real-time system that runs on commodity hardware and can be optimized for edge deployment.

## Locked Scope Decisions (v2)
1. MVP gesture set: 6 gestures.
2. No external hand-landmark model baseline; core vision stack implemented by the team.
3. Raspberry Pi deployment target: minimum 15 FPS end-to-end.
4. Eye tracking is stretch scope only after core project milestones are complete.

## Class Topics Most Relevant to This Project
Based on the CS4100 course page and notes links, these are the most useful topics to prioritize:

1. Feature Extraction and Vector Representations
- Why it matters: your pipeline depends on turning raw webcam frames/video clips into robust gesture features (contours, landmarks, motion descriptors, or learned embeddings).
- Course link: https://rajagopalvenkat.com/teaching/resources/AI/ch7.html

2. Machine Learning Classifiers (Naive Bayes, Linear Classifiers, SVM, Logistic Regression)
- Why it matters: strong baselines for static gesture classification before moving to deeper models.
- Course link: https://rajagopalvenkat.com/teaching/resources/AI/ch7.html

3. Gradient Descent and Model Training
- Why it matters: core training loop for tuning classifiers/neural models and understanding optimization tradeoffs under latency constraints.
- Course link: https://rajagopalvenkat.com/teaching/resources/AI/ch7.html

4. Neural Networks
- Why it matters: likely best-performing option for image-based gesture recognition; needed for compact but accurate models for Raspberry Pi.
- Course link: https://rajagopalvenkat.com/teaching/resources/AI/ch7.html

5. Markov Models and Hidden Markov Models (HMMs)
- Why it matters: directly useful for motion-based gesture recognition over time (sequence modeling, smoothing, temporal decoding).
- Course link: https://rajagopalvenkat.com/teaching/resources/AI/ch6.html

6. Computer Vision / Deep Q Nets lecture deck (CV portion)
- Why it matters: practical framing of image representation and deep feature extraction for vision tasks.
- Course link: https://rajagopalvenkat.com/teaching/lectures/CV_DQN/

Lower priority for this project: adversarial search, CSPs, and most RL-specific control topics unless you explicitly add adaptive interaction or policy-learning behavior.

## Suggested Implementation Plan (Module-Based)

### Module A: Data Strategy and Dataset Pipeline
- Define gesture vocabulary:
  - Static: play/pause, volume up/down.
  - Motion: next/previous (e.g., swipe left/right).
- Build data pipeline for:
  - Public seed data (if available).
  - Team-collected videos across lighting, skin tones, camera distances, backgrounds.
- Create train/val/test split protocol and labeling format.

Deliverable: reproducible dataset manifest + scripts in repo.

### Module B: Vision Front-End (Detection + Preprocessing)
- Frame acquisition and resizing.
- Hand region localization (team-built implementation only).
- Preprocessing: normalization, background handling, temporal windowing for motion gestures.

Deliverable: reusable preprocessing package with unit-testable components.

### Module C: Static Gesture Model Track
- Baseline model: engineered features + classical classifier.
- Improved model: compact neural model.
- Compare accuracy/latency/robustness.

Deliverable: benchmark table and confusion matrices for static classes.

### Module D: Motion Gesture Model Track
- Start with temporal feature baseline + HMM/sequence classifier.
- Add windowed smoothing/post-processing to reduce false triggers.
- Define minimum confidence and motion duration thresholds.

Deliverable: motion recognition metrics and demo clips.

### Module E: Command Engine and UX Safety Layer
- Map predictions to media commands.
- Add debounce, cooldown, and state gating (avoid repeated accidental triggers).
- Keep keyboard/mouse fallback for safety during demos.

Deliverable: command middleware with configurable gesture-command mapping.

### Module F: Real-Time App Integration
- End-to-end live inference loop.
- Hook into media control backend (OS keystrokes or media API).
- Overlay diagnostics (predicted class/confidence/fps).

Deliverable: desktop demo app.

### Module G: Raspberry Pi Deployment and Optimization
- Port inference pipeline to Raspberry Pi + camera.
- Profile FPS, memory, CPU usage, thermal stability.
- Optimize model/input resolution/pipeline scheduling to hit >=15 FPS usable real-time performance.

Deliverable: Pi demo + performance report.

### Module H: Evaluation, Reporting, and Presentation Assets
- Quantitative: accuracy, precision/recall/F1 per class, confusion matrices, latency/fps.
- Qualitative: usability trials across 3-5 users and environments.
- Prepare final demo video and reproducibility documentation.

Deliverable: figures/tables for final summary + presentation-ready visual assets.

## Timeline Aligned to Course Milestones (Spring 2026)

1. Feb 11 to Feb 23, 2026
- Finalize gesture set, dataset schema, module ownership.
- Build baseline preprocessing + static classifier prototype.

2. First check-in target (last week of February 2026)
- Show working static gesture pipeline end-to-end.
- Show early motion-gesture baseline design and data collection progress.

3. Mar 1 to Mar 23, 2026
- Implement motion gesture recognition + command safety layer.
- Integrate app-level real-time control.

4. Second check-in target (last week of March 2026)
- Show integrated static + motion system with quantitative metrics.
- Show first Raspberry Pi inference measurements.

5. Mar 30 to Apr 8, 2026
- Optimize Pi performance and robustness.
- Freeze experiments and produce final plots/tables.
- Finalize slides and demo video (slides due Apr 8, 2026).

6. Apr 10 to Apr 17, 2026
- Presentation delivery window (Apr 10/14/17, 2026).
- Final deliverables due Apr 17, 2026.

## Equitable 4-Person Work Breakdown
Use lead + secondary ownership to keep workloads fair and visible in GitHub history.

1. Member 1 (Data + Evaluation Lead)
- Lead: Module A and Module H.
- Secondary: test support for Module C/D.
- Main outputs: dataset tooling, experiment tracking, final metrics package.

2. Member 2 (Vision Pipeline Lead)
- Lead: Module B and support Module G optimization.
- Secondary: live pipeline integration help in Module F.
- Main outputs: preprocessing stack, efficient frame pipeline.

3. Member 3 (Modeling Lead)
- Lead: Module C and Module D.
- Secondary: confidence calibration with Module E.
- Main outputs: static + motion models, ablation comparisons.

4. Member 4 (Systems + Deployment Lead)
- Lead: Module E, Module F, and Module G integration.
- Secondary: presentation demo tooling for Module H.
- Main outputs: command engine, app integration, Raspberry Pi deployment.

Shared responsibilities for equity:
- Every member owns at least one model/data/system code path in commits.
- Pair-review all PRs (no solo merges for core modules).
- Rotate weekly “integration captain” to prevent siloing.
- Track contributions via GitHub Projects tasks tied to module deliverables.

## Follow-Up Questions to Resolve Early
1. Which six gestures will be locked for MVP evaluation and demo?
2. What test environments are required for validation (lighting, distance, background, user variation)?
3. What fallback behavior should trigger if FPS drops below 15 on Raspberry Pi?
4. What exact criteria unlock eye-tracking stretch work (for example, all core modules complete + Pi target met)?

## Additional Items Worth Adding to This Initial Plan
- Risk register (dataset bias, lighting sensitivity, false positives, Pi performance risk).
- Definition of done per module (metrics + demo criteria).
- Experiment log template (hyperparameters, dataset version, outcomes).
- TA check-in agenda template with blockers/asks.
- Final demo script and fallback plan in case of live hardware failure.
