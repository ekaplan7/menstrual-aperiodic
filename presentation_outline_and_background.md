# Hormonal Contraception & the Resting Brain
### Presentation outline, speaker notes, and literature background

A working reference for assembling the talk (mixing these slides with the Canva slides). Part 1 is a slide-by-slide outline with brief speaker notes. Part 2 is background on how hormones/HC affect the **aperiodic** and **periodic** EEG, with sources. Part 3 is how our findings sit against that literature (read this before presenting).

---

## PART 1 — Slide outline + speaker notes

Numbers match the current `hc_aperiodic_deck.pptx`. Speaker notes are 2–4 talking points each; trim/expand as you slot Canva slides in.

**1 · Title — Hormonal contraception & the resting brain**
- Set the frame: does current HC use leave a fingerprint in resting EEG?
- Two signals from one recording — the aperiodic 1/f background and the oscillatory alpha rhythm.
- Dataset ds007615, 69 women, eyes-closed + eyes-open.

**2 · A near-universal exposure, an understudied brain**
- Hundreds of millions of women use hormonal contraception daily — it's among the most-used medications on earth.
- Yet its effects on brain excitability are thinly characterized and inconsistent.
- Motivation: a large population, a small and mixed evidence base.

**3 · Sex hormones tune cortical excitability**
- Estrogen is broadly excitatory (↑ glutamate/NMDA, ↓ GABA); progesterone (via allopregnanolone) is inhibitory (positive GABA-A modulator).
- HC suppresses the natural cycle and replaces it with a steady exogenous hormone profile.
- Prediction: HC should reshape the cortical excitation–inhibition (E/I) balance — and the aperiodic EEG is a candidate readout.

**4 · Reading excitability from the EEG spectrum**
- The power spectrum = an aperiodic 1/f component (exponent + offset) plus periodic peaks (alpha).
- specparam separates the two per channel, so aperiodic and oscillatory effects don't contaminate each other.
- This separation is the methodological backbone of the whole talk.

**5 · The question & the design**
- Can resting EEG decode current HC use, and is it neural signal rather than confounds?
- Four LOSO models: confounds-only (≈ chance), EEG-only (reference), EEG+confounds, EEG-residualized (the clean test).
- Residualization is fold-wise (train-only) → leakage-safe; significance by permutation.

**6 · Sample**
- 69 women: 37 current HC users vs 32 non-current (21 past, 11 never).
- Decode target = current vs non-current use.

**7 · Variables we controlled for**
- Age, education, depression (BDI), alcohol, daily nicotine, menstrual cycle phase.
- These are the plausible confounds of a hormonal→EEG association.

**8 · Methods**
- Features: exponent + offset × EC/EO × 64 channels (256 features).
- L2 logistic regression, leave-one-subject-out CV.
- Confounds regressed out of every feature within each training fold; 500-run permutation test.

**9 · Confounds don't separate the groups**
- Groups broadly comparable on the measured confounds.
- Confounds-only classifier ≈ chance (AUC ≈ 0.51) — they don't carry the HC signal.

**10 · The menstrual cycle-phase question**
- Phase barely predicts HC use, so it isn't a classic confound.
- BUT phase data is missing for ~28%, and missingness tracks HC use (indicator alone ≈ 0.66 AUC).
- Fix: mean-impute phase, do NOT use a missingness indicator (it leaks the label); back up with complete-case analysis.

**11 · Result: the aperiodic decode survives full confound control**
- Confounds-only ≈ 0.51; EEG-only ≈ 0.70; EEG | confounds ≈ 0.71.
- Permutation p ≈ 0.02 (complete-case n = 50). The decode is not a confound artifact.

**12 · Where the aperiodic signal lives**
- Coefficients spread across the scalp — a distributed pattern, not one electrode.
- Exponent + offset in both EC and EO contribute; the scalp-average exponent moves only modestly.

**13 · A hormonal lead: estrogen-containing contraceptives** *(exploratory)*
- Combined (estrogen) vs progestin-only: markedly flatter exponent (d ≈ 1.0), age-adjusted.
- Direction fits estrogen's pro-excitatory action (flatter slope = more excitation).
- Effect is diffuse, not focal (per-channel all same direction; 1 channel survives FDR).

**14 · The periodic side: alpha power is elevated in HC users** *(exploratory)*
- Current HC users show higher oscillatory alpha power (d ≈ 0.76 eyes-open; robust to age + nicotine).
- Survives 1/f removal → a genuine oscillation, and it's alpha-specific (not theta/beta).

**15 · A double dissociation**
- HC use (vs non-use) → moves alpha oscillations (+ the distributed aperiodic pattern).
- Formulation (estrogen vs progestin) → moves the aperiodic exponent, not oscillations.
- Caveat: formulation comparison underpowered (16 vs 21).

**16 · Takeaways**
- Two-part resting-EEG signature of current HC use: distributed aperiodic pattern (AUC ≈ 0.70) + elevated alpha (d ≈ 0.76).
- Not explained by age, nicotine, alcohol, education, depression, or cycle phase (survives residualization + permutation).
- Double dissociation of the two hormonal contrasts.
- Limits: single dataset, all-female, cross-sectional; secondary analyses exploratory/uncorrected; cycle-phase underpowered.

---

## PART 2 — Literature background: hormones, contraception, and the EEG spectrum

### 2.1 The aperiodic 1/f slope as a marker of E/I balance
The slope (exponent) of the aperiodic 1/f component is now widely used as a non-invasive proxy for the cortical **excitation–inhibition (E/I) balance**: modeling and pharmacology work show that **more inhibition steepens** the slope and **more excitation flattens** it. This is the interpretive anchor for reading hormone effects off the spectrum. E/I-linked aperiodic exponent has since been related to cognition and gene expression (e.g., in temporal-lobe epilepsy) and to perceptual/temporal processing.
- Excitation/inhibition & aperiodic exponent (epilepsy, high-density EEG): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11272395/
- Aperiodic EEG & visual temporal processing: https://pmc.ncbi.nlm.nih.gov/articles/PMC11450528/

### 2.2 Sex hormones modulate cortical excitability (mechanism)
Estradiol is broadly **excitatory** — it augments NMDA/glutamatergic transmission and reduces GABA production. Progesterone, via its metabolite **allopregnanolone**, is **inhibitory** — a positive allosteric modulator of GABA-A receptors with benzodiazepine-like effects. TMS and psychophysics studies show intracortical excitability tracks these hormones across the natural cycle.
- Estradiol/progesterone modulate intracortical excitability (TMS, Sci Rep 2020): https://www.nature.com/articles/s41598-020-79389-6
- Menstrual cycle effects on cortical excitability (Neurology): https://www.neurology.org/doi/10.1212/wnl.53.9.2069
- Ovarian hormones & GABAergic regulation of neural excitability: https://www.ncbi.nlm.nih.gov/pubmed/25936144

### 2.3 Periodic (alpha) effects of cycle, hormones, and contraceptives
The oscillatory literature is real but **mixed**:
- **Alpha frequency (IAF / peak):** relates to cycle phase and estradiol — individual alpha frequency tends to be **highest in the luteal phase, lowest in late-follicular**, and **correlates negatively with estradiol**; oral-contraceptive use is also associated with alpha frequency. (Haarmann/Bazanova-type findings.)
  - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4152552/
- **Alpha power:** inconsistent across studies — some report higher alpha in follicular vs luteal, others the reverse for upper-alpha; salivary progesterone has been positively associated with alpha peak frequency and upper-alpha (α2) power.
  - https://link.springer.com/article/10.1134/S0362119714020030
- **Sex differences may dominate:** at least one recent study finds resting-EEG **sex differences are more prominent than menstrual-cycle effects** in young adults.
  - https://www.frontiersin.org/journals/endocrinology/articles/10.3389/fendo.2026.1785349/full

### 2.4 The direct precedent — and its null result *(read this)*
A 2024 paper specifically examined **female hormonal status, alpha oscillations, and aperiodic features of resting-state EEG** across naturally-cycling, oral-contraceptive, and hormonal-IUD groups (with salivary estradiol/progesterone/testosterone). Its headline finding was **null**: hormonal status was **not** related to the 1/f slope, alpha power, or peak frequency, and EEG parameters did not track hormone levels. This is the closest published precedent to our questions.
- https://www.sciencedirect.com/science/article/abs/pii/S0167876024000163

**Note — this is a *separate* study, not our dataset.** ds007615 is *"LDAEP and resting-state EEG in healthy women"* (Normannseth, Hatlestad-Hall, Rygvold, Hadzic & Andersson, 2025, *Frontiers in Human Neuroscience*), whose original focus was HC use and central **serotonergic** activity via the loudness-dependence of auditory evoked potentials — **not** aperiodic or oscillatory resting EEG. So the aperiodic/periodic questions in this talk are a **novel secondary analysis** of the dataset, not a re-run of a prior aperiodic study.
- Source study / how to cite the data: Normannseth et al. (2025), *Front. Hum. Neurosci.* 19:1647425 — https://doi.org/10.3389/fnhum.2025.1647425

Broader context: a 60-year systematic review of HC and the brain concludes the field is heterogeneous and understudied, with few robust, replicated EEG effects.
- https://www.sciencedirect.com/science/article/abs/pii/S0091302222000747

Broader context: a 60-year systematic review of HC and the brain concludes the field is heterogeneous and understudied, with few robust, replicated EEG effects.
- https://www.sciencedirect.com/science/article/abs/pii/S0091302222000747

---

## PART 3 — How our findings sit against the literature (framing)

1. **Novel angle on this dataset + a diverging precedent.** ds007615 was collected for a serotonergic (LDAEP) question, so aperiodic/oscillatory HC effects are unexamined here — this is a first look. The closest published precedent (the 2024 aperiodic study on a *different* sample) reported a univariate null (no hormonal relationship to 1/f slope, alpha power, or peak frequency). Our contribution is methodological: a **multivariate, confound-controlled decode** (64-channel pattern, not a scalp average) plus **subgroup contrasts** (formulation) surface structure a univariate test can miss. Frame our results as *"a distributed, confound-controlled re-analysis that recovers signal univariate tests overlook"* — consistent with the null precedent, not a bare contradiction of it.

2. **Our direction is mechanistically consistent.** Estrogen-containing formulation → **flatter** exponent = more excitation-dominated, matching estrogen's known excitatory pharmacology and the E/I→slope mapping (2.1–2.2). This is the most defensible novel claim, but it's underpowered (16 vs 21) and needs replication.

3. **Our alpha result aligns with the "hormones touch alpha" strand** (2.3) but not the precedent null — so present it as exploratory and note the mixed prior literature (power findings genuinely conflict across studies).

4. **Our cycle-phase nulls are consistent with the literature's weak/inconsistent cycle effects** (2.3–2.4) and the "sex > cycle" finding. Honest and easy to defend.

5. **Recommended hedging language for the talk:** lead the confirmatory story with the confound-controlled decode (rigorous, permutation-tested); present formulation, alpha, and the dissociation explicitly as *exploratory leads for replication.* Acknowledge the precedent null directly — it makes the talk stronger, not weaker.

**Citation:** cite the dataset as Normannseth et al. (2025), *Front. Hum. Neurosci.* 19:1647425 (OpenNeuro ds007615), and position this talk as a **novel secondary analysis** of resting-state aperiodic + oscillatory EEG. Cite the 2024 aperiodic study as the nearest precedent/contrast.
