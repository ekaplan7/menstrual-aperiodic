"""
Neural Timescales HC Decoding — Confound-Controlled
Aligned to classification.ipynb (aperiodic exponent + offset decode)

═══════════════════════════════════════════════════════════════════════════════
ALIGNMENT SUMMARY
───────────────────────────────────────────────────────────────────────────────
classification.ipynb                     → THIS SCRIPT
─────────────────────────────────────────────────────────────────────────────
aperiodic_per_subject_channel.csv        → knee_frequencies.csv (knee_freq_hz)
                                           acf_timescales.csv (ACW_0/50/1e)
PARAMS = ['exponent', 'offset']          → METRICS (one metric per run)
feature matrix: exponent+offset ×        → feature matrix: metric × EC+EO ×
  EC+EO × 64ch = 256 features              64ch = 128 features (per metric)
4-model LOSO L2 logistic (C=0.1)        → identical 4-model LOSO L2 logistic
N_PERM=500 permutation test              → N_PERM=500
Phase-restricted sensitivity (n~50)      → identical phase-restricted sensitivity
hc_covariate_control_results.txt         → results_{metric}.txt
hc_coef_topomap.png (1×4 panel)         → coef_topomap_{metric}.png (1×2 panel)
derivatives/preproc/ml/                  → derivatives/models/models_timescales_2s/
                                           (or models_timescales_4s/)
═══════════════════════════════════════════════════════════════════════════════

KEY DESIGN DECISIONS (to match classification.ipynb exactly):
  • EC and EO are NOT analysed separately — they are stacked into ONE unified
    feature matrix per metric, exactly as exponent_ec_* + exponent_eo_* +
    offset_ec_* + offset_eo_* are all combined in classification.ipynb.
  • Each of the 4 timescale metrics is run INDEPENDENTLY (not all stacked),
    giving one complete decode per metric for direct comparison.
  • Covariates: age, daily_nicotine, alcohol_units, edu, bdi_total,
    mens_phase=ovulatory, mens_phase=luteal  (same 7 as classification.ipynb).
  • Cycle-phase missingness: mean-imputed (NO indicator), same rationale as
    classification.ipynb (missingness proxy AUC ~0.66 → leaky if used).
  • LOSO CV, L2 logistic C=0.1, 500 permutations — all identical.

INPUTS:
  derivatives/preproc/specparam/knee_frequencies.csv      (2s window)
  derivatives/preproc/specparam/acf_timescales.csv        (2s window)
  ds007615-download/participants.tsv
  ds007615-download/phenotype/lifestyle.tsv
  ds007615-download/phenotype/bdi.tsv
  ds007615-download/phenotype/hc_usage.tsv

OUTPUTS (derivatives/models/models_timescales_2s/):
  results_{metric}.txt          per-metric decode results (mirrors hc_covariate_control_results.txt)
  coef_topomap_{metric}.png     logistic coefficient topomap
  {metric}_model.pkl            serialised fitted model (for reloading in figures script)
  summary_timescales.csv        one-row-per-metric summary table
  summary_timescales.txt        human-readable summary

Switch WINDOW_SIZE = '4s' to use preproc_long data → outputs go to models_timescales_4s/.

Requires: pandas numpy scikit-learn scipy matplotlib mne
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG  ← edit here
# ============================================================
WINDOW_SIZE = '4s'   # '2s' → preproc (aligned to aperiodic); '4s' → preproc_long

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
BIDS_ROOT  = os.path.join(BASE_DIR, 'ds007615-download')
DERIV_ROOT = os.path.join(BASE_DIR, 'derivatives')

if WINDOW_SIZE == '2s':
    SPECPARAM_DIR = os.path.join(DERIV_ROOT, 'preproc', 'specparam')
    KNEE_CSV      = os.path.join(SPECPARAM_DIR, 'knee_frequencies.csv')
    ACF_CSV       = os.path.join(SPECPARAM_DIR, 'acf_timescales.csv')
    OUT_SUBDIR    = 'models_timescales_2s'
elif WINDOW_SIZE == '4s':
    SPECPARAM_DIR = os.path.join(DERIV_ROOT, 'preproc_long', 'specparam')
    KNEE_CSV      = os.path.join(SPECPARAM_DIR, 'knee_frequencies_master_new.csv')
    ACF_CSV       = os.path.join(SPECPARAM_DIR, 'acf_timescales_new.csv')
    OUT_SUBDIR    = 'models_timescales_4s'
else:
    raise ValueError("WINDOW_SIZE must be '2s' or '4s'")

OUT_DIR       = os.path.join(DERIV_ROOT, 'models', OUT_SUBDIR)
PARTICIPANTS  = os.path.join(BIDS_ROOT, 'participants.tsv')
PHENOTYPE_DIR = os.path.join(BIDS_ROOT, 'phenotype')
os.makedirs(OUT_DIR, exist_ok=True)

# Metrics to decode — each run independently, EC+EO unified per metric
# Naming mirrors PARAMS = ['exponent', 'offset'] in classification.ipynb
METRICS = [
    'knee_freq_hz',   # from knee_frequencies CSV  (specparam knee mode)
    'ACW_0_ms',       # from acf_timescales CSV
    'ACW_50_ms',      # from acf_timescales CSV
    'ACW_1e_ms',      # from acf_timescales CSV
]

CONDITIONS   = ['ec', 'eo']   # stacked into unified matrix (not run separately)
C_REG        = 0.1            # identical to classification.ipynb
N_PERM       = 500            # identical to classification.ipynb
RANDOM_STATE = 42
rng = np.random.RandomState(RANDOM_STATE)

# Quality filter for specparam knee fits (mirrors neural_timescales_mechanisms.py)
# Fits with r²<0.95 are excluded before building the feature matrix.
# NOTE: knee_freq_hz has ~16% NaN from non-converging fits on scalp EEG.
# This filter addresses an additional ~3% of low-quality fits.
# Poor classification AUC for knee (~0.50) reflects data quality and
# signal absence at the per-channel level — NOT a code bug.
# For group-level analyses (ANOVA, OLS), use neural_timescales_mechanisms.py
# which averages across channels first, making NaN handling robust.
R2_THRESH = 0.95

print(f'WINDOW_SIZE : {WINDOW_SIZE}')
print(f'OUT_DIR     : {OUT_DIR}')


# ============================================================
# PIPELINE HELPERS  — identical machinery to classification.ipynb
# ============================================================
class ConfoundRegressor(BaseEstimator, TransformerMixin):
    """Regress confound columns OUT of signal columns (fold-wise, leakage-safe).
    Confounds are assumed to be the LAST n_confounds columns of X.
    Mirrors the ConfoundRegressor in classification.ipynb exactly."""
    def __init__(self, n_confounds):
        self.n_confounds = n_confounds

    def fit(self, X, y=None):
        Xs, C = X[:, :-self.n_confounds], X[:, -self.n_confounds:]
        self.c_mean_ = C.mean(0)
        design = np.column_stack([np.ones(len(C)), C - self.c_mean_])
        self.beta_ = np.linalg.lstsq(design, Xs, rcond=None)[0]
        return self

    def transform(self, X):
        Xs, C = X[:, :-self.n_confounds], X[:, -self.n_confounds:]
        design = np.column_stack([np.ones(len(C)), C - self.c_mean_])
        return Xs - design @ self.beta_


def _logistic():
    return LogisticRegression(penalty='l2', C=C_REG, max_iter=2000, solver='liblinear')

def base_pipe():
    """Impute → scale → logistic  (no confound removal)."""
    return Pipeline([('imp', SimpleImputer()), ('sc', StandardScaler()), ('lr', _logistic())])

def resid_pipe(nc):
    """Impute → confound-residualize → scale → logistic."""
    return Pipeline([('imp', SimpleImputer()),
                     ('cr', ConfoundRegressor(nc)),
                     ('sc', StandardScaler()),
                     ('lr', _logistic())])

def loso_auc(pipe, X, y):
    """Leave-one-subject-out AUC + balanced accuracy."""
    proba = cross_val_predict(pipe, X, y, cv=LeaveOneOut(), method='predict_proba')[:, 1]
    auc  = roc_auc_score(y, proba)
    bacc = balanced_accuracy_score(y, (proba > 0.5).astype(int))
    return auc, bacc

def perm_p(nc, X, y, n_perm=N_PERM):
    """Permutation p-value on the confound-residualized model.
    Mirrors perm_p() in classification.ipynb exactly."""
    obs, _ = loso_auc(resid_pipe(nc), X, y)
    null = np.empty(n_perm)
    for i in range(n_perm):
        yp = rng.permutation(y)
        pr = cross_val_predict(resid_pipe(nc), X, yp,
                               cv=LeaveOneOut(), method='predict_proba',
                               n_jobs=1)[:, 1]
        null[i] = roc_auc_score(yp, pr)
    p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    return obs, p, null.mean()


# ============================================================
# LOAD RAW TIMESCALE DATA
# ============================================================
df_knee = pd.read_csv(KNEE_CSV, dtype={'subject': int})
df_acf  = pd.read_csv(ACF_CSV,  dtype={'subject': int})

# Apply quality filter to knee data (set low-r² fits to NaN before feature matrix)
if 'r_squared' in df_knee.columns:
    n_bad = (df_knee['r_squared'] < R2_THRESH).sum()
    n_orig_nan = df_knee['knee_freq_hz'].isna().sum()
    df_knee.loc[df_knee['r_squared'] < R2_THRESH, 'knee_freq_hz'] = np.nan
    n_total_nan = df_knee['knee_freq_hz'].isna().sum()
    print(f'[knee r²-filter] {n_bad} entries below r²={R2_THRESH} set to NaN '
          f'(orig NaN={n_orig_nan} → total NaN={n_total_nan} / {len(df_knee)} = '
          f'{100*n_total_nan/len(df_knee):.1f}%)')

# Map each metric to its source dataframe and value column
METRIC_SOURCE = {
    'knee_freq_hz': (df_knee, 'knee_freq_hz'),
    'ACW_0_ms':     (df_acf,  'ACW_0_ms'),
    'ACW_50_ms':    (df_acf,  'ACW_50_ms'),
    'ACW_1e_ms':    (df_acf,  'ACW_1e_ms'),
}

print(f'\nknee_frequencies: {df_knee.shape}  subjects={df_knee.subject.nunique()}')
print(f'acf_timescales  : {df_acf.shape}   subjects={df_acf.subject.nunique()}')


# ============================================================
# LOAD COVARIATES — identical to classification.ipynb
# Covariates: age, daily_nicotine, alcohol_units, edu, bdi_total,
#             mens_phase=ovulatory, mens_phase=luteal  (7 total)
# ============================================================
def _load_tsv(path):
    d = pd.read_csv(path, sep='\t', na_values='n/a')
    d['subject'] = d['participant_id'].str.replace('sub-', '', regex=False).astype(int)
    return d.set_index('subject')

parts = _load_tsv(PARTICIPANTS)
life  = _load_tsv(os.path.join(PHENOTYPE_DIR, 'lifestyle.tsv'))
bdi   = _load_tsv(os.path.join(PHENOTYPE_DIR, 'bdi.tsv'))
hc    = _load_tsv(os.path.join(PHENOTYPE_DIR, 'hc_usage.tsv'))

COV_LABELS = ['age', 'daily_nicotine', 'alcohol_units', 'edu', 'bdi_total',
              'mens_phase=ovulatory', 'mens_phase=luteal']


def build_covariates(subjects):
    """Build covariate matrix Z and target y.
    Mirrors build_covariates() in neural_timescales_modeling.ipynb and the
    covariate block in classification.ipynb. Cycle phase mean-imputed (no indicator)."""
    idx = pd.Index(subjects)
    y    = (parts.loc[idx, 'group'] == 1).astype(int).values   # 1 = current HC user
    age  = parts.loc[idx, 'age'].astype(float).values
    nic  = (life.loc[idx, 'daily_nicotine'] == 1).astype(float).values
    alc  = life.loc[idx, 'alcohol_units'].astype(float).values
    edu  = parts.loc[idx, 'edu'].astype(float).values
    bdit = bdi.loc[idx, 'bdi_total'].astype(float).values
    mphase    = hc.loc[idx, 'mens_phase'].astype(float).values
    have_phase = ~np.isnan(mphase)
    ovu = np.where(have_phase, (mphase == 2).astype(float), np.nan)  # NaN → mean-imputed in pipe
    lut = np.where(have_phase, (mphase == 3).astype(float), np.nan)
    Z   = np.column_stack([age, nic, alc, edu, bdit, ovu, lut])
    return y, Z, have_phase, mphase


# ============================================================
# TOPOMAP HELPER
# ============================================================
def save_coef_topomap(coef_by_feat, feat_names, metric, channels, out_path):
    """
    1×2 coefficient topomap (EC | EO) for one metric.
    Red = predicts current HC use. Mirrors hc_coef_topomap.png structure.
    """
    try:
        import mne
        info = mne.create_info(list(channels), sfreq=1.0, ch_types='eeg')
        info.set_montage(mne.channels.make_standard_montage('standard_1005'),
                         on_missing='ignore')
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for ax, acq in zip(axes, CONDITIONS):
            vals = np.array([coef_by_feat.get(f'{metric}_{acq}_{ch}', np.nan)
                             for ch in channels])
            vmax = np.nanmax(np.abs(vals)) or 1.0
            try:
                im, _ = mne.viz.plot_topomap(
                    vals, info, axes=ax, show=False,
                    cmap='RdBu_r', vlim=(-vmax, vmax), contours=0
                )
                ax.set_title(f'{acq.upper()}', fontsize=12)
            except Exception as e:
                ax.text(0.5, 0.5, str(e), transform=ax.transAxes, ha='center', va='center')
        fig.suptitle(f'Logistic coefficients: {metric}\n'
                     f'Red = predicts current HC use  |  EEG-only model',
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(out_path, dpi=140, bbox_inches='tight')
        plt.close(fig)
        return f'saved {os.path.basename(out_path)}'
    except Exception as e:
        return f'topomap skipped ({type(e).__name__}: {e})'


# ============================================================
# MAIN LOOP — one complete decode per metric
# ============================================================
summary_rows = []

for metric in METRICS:
    print(f'\n{"="*60}')
    print(f'METRIC: {metric}  ({WINDOW_SIZE} window)')
    print(f'{"="*60}')

    df_src, val_col = METRIC_SOURCE[metric]

    # ── Build unified EC+EO feature matrix (mirrors classification.ipynb) ──────
    # classification.ipynb: pivot_table(values=['exponent','offset'], columns=['acq','channel'])
    # Here: same pivot but values=[metric] → columns = (acq, channel) with naming {metric}_{acq}_{ch}
    df_m = df_src[df_src['acq'].isin(CONDITIONS)][['subject', 'acq', 'channel', val_col]].copy()
    wide = df_m.pivot_table(index='subject', columns=['acq', 'channel'], values=val_col)
    wide.columns = [f'{metric}_{acq}_{ch}' for (acq, ch) in wide.columns]
    wide = wide.sort_index()

    subjects   = wide.index.tolist()   # list of int subject IDs
    X_eeg      = wide.values
    feat_names = list(wide.columns)
    n_sub      = len(subjects)
    nan_pct    = float(np.isnan(X_eeg).mean() * 100)
    n_feats    = X_eeg.shape[1]

    print(f'Feature matrix: {n_sub} subjects × {n_feats} features')
    print(f'  = 1 metric × {len(CONDITIONS)} conditions × ~64 channels')
    print(f'  NaN% = {nan_pct:.1f}  (mean-imputed inside each CV fold)')

    # ── Covariates & target ───────────────────────────────────────────────────
    y, Z, have_phase, mphase = build_covariates(subjects)
    n_conf = Z.shape[1]
    n_hc   = int(y.sum())
    print(f'  n={n_sub}  HC={n_hc}  non-HC={n_sub - n_hc}')

    # ── 4-model LOSO (identical to classification.ipynb) ─────────────────────
    print('  Fitting models...')
    auc_eeg,  bacc_eeg  = loso_auc(base_pipe(),     X_eeg,                 y)
    auc_cov,  bacc_cov  = loso_auc(base_pipe(),     Z,                     y)
    auc_both, bacc_both = loso_auc(base_pipe(),     np.hstack([X_eeg, Z]), y)
    auc_res,  p_res, null_mean = perm_p(n_conf,     np.hstack([X_eeg, Z]), y)
    _, bacc_res = loso_auc(resid_pipe(n_conf),      np.hstack([X_eeg, Z]), y)

    print(f'\n  [full n={n_sub}]')
    print(f'  covariates only          AUC {auc_cov:.3f}  bal.acc {bacc_cov:.3f}')
    print(f'  EEG only                 AUC {auc_eeg:.3f}  bal.acc {bacc_eeg:.3f}')
    print(f'  EEG + covariates         AUC {auc_both:.3f}  bal.acc {bacc_both:.3f}')
    print(f'  EEG | covariates (resid) AUC {auc_res:.3f}  bal.acc {bacc_res:.3f}  perm p={p_res:.4f}')

    # ── Phase-restricted sensitivity (identical to classification.ipynb) ──────
    m       = have_phase
    n_phase = int(m.sum())
    idx_m   = [i for i, s in enumerate(subjects) if m[i]]
    Xr = X_eeg[m]
    yr = y[m]
    Zr = np.column_stack([
        np.array([parts.loc[s, 'age']                     for s in subjects], dtype=float)[m],
        np.array([(life.loc[s, 'daily_nicotine'] == 1)    for s in subjects], dtype=float)[m],
        np.array([life.loc[s, 'alcohol_units']            for s in subjects], dtype=float)[m],
        np.array([parts.loc[s, 'edu']                     for s in subjects], dtype=float)[m],
        np.array([bdi.loc[s, 'bdi_total']                 for s in subjects], dtype=float)[m],
        (mphase[m] == 2).astype(float),
        (mphase[m] == 3).astype(float),
    ])
    auc_eeg_r, _ = loso_auc(base_pipe(),                  Xr,                    yr)
    auc_cov_r, _ = loso_auc(base_pipe(),                  Zr,                    yr)
    auc_res_r, p_res_r, _ = perm_p(Zr.shape[1], np.hstack([Xr, Zr]),             yr)

    print(f'\n  [restricted n={n_phase}] EEG {auc_eeg_r:.3f} | '
          f'cov {auc_cov_r:.3f} | EEG|cov {auc_res_r:.3f} perm p={p_res_r:.4f}')

    # ── Coefficient topomap ───────────────────────────────────────────────────
    pipe = base_pipe()
    pipe.fit(X_eeg, y)
    coef_by_feat  = dict(zip(feat_names, pipe.named_steps['lr'].coef_.ravel()))
    channels_all  = sorted(df_m['channel'].unique())
    topo_path     = os.path.join(OUT_DIR, f'coef_topomap_{metric}.png')
    topo_note     = save_coef_topomap(coef_by_feat, feat_names, metric, channels_all, topo_path)
    print(f'  {topo_note}')

    # ── Save model .pkl ───────────────────────────────────────────────────────
    pkl_path = os.path.join(OUT_DIR, f'{metric}_model.pkl')
    with open(pkl_path, 'wb') as fh:
        pickle.dump({
            'pipe':          pipe,
            'feat_names':    feat_names,
            'coef_by_feat':  coef_by_feat,
            'channels':      channels_all,
            'metric':        metric,
            'window_size':   WINDOW_SIZE,
            'auc_eeg':       auc_eeg,
            'auc_res':       auc_res,
            'p_res':         p_res,
        }, fh)
    print(f'  saved {os.path.basename(pkl_path)}')

    # ── Save text results — mirrors hc_covariate_control_results.txt format ──
    txt_path = os.path.join(OUT_DIR, f'results_{metric}.txt')
    cov_str  = ', '.join(COV_LABELS)
    sig_str  = ('***' if p_res < 0.001 else '**' if p_res < 0.01
                else '*' if p_res < 0.05 else '+' if p_res < 0.10 else 'n.s.')
    with open(txt_path, 'w') as fh:
        fh.write(f'HC-use decoding — neural timescale: {metric}  ({WINDOW_SIZE} window)\n')
        fh.write('ALIGNED TO: classification.ipynb (aperiodic exponent + offset decode)\n')
        fh.write('=' * 62 + '\n')
        fh.write(f'n subjects    = {n_sub} | current HC={n_hc}  non-users={n_sub - n_hc}\n')
        fh.write(f'metric        = {metric}\n')
        fh.write(f'features      = {metric} × {len(CONDITIONS)} conditions × '
                 f'{len(channels_all)} channels = {n_feats} total\n')
        fh.write(f'  [aperiodic equiv: exponent+offset × EC+EO × 64ch = 256 features]\n')
        fh.write(f'NaN%          = {nan_pct:.1f}  (mean-imputed inside each CV fold)\n')
        fh.write(f'covariates    = {cov_str}\n')
        fh.write('cycle phase   = mean-imputed on full sample (no missingness indicator:\n')
        fh.write('  indicator alone decodes HC use at AUC ~0.66 → leaky label proxy)\n')
        fh.write(f'CV            = leave-one-subject-out | L2 logistic C={C_REG}\n')
        fh.write(f'permutations  = {N_PERM}\n\n')
        fh.write(f'{"model":<30}{"AUC":>7}{"bal.acc":>9}\n')
        fh.write('-' * 46 + '\n')
        fh.write(f'{"covariates only":<30}{auc_cov:>7.3f}{bacc_cov:>9.3f}\n')
        fh.write(f'{"EEG only":<30}{auc_eeg:>7.3f}{bacc_eeg:>9.3f}\n')
        fh.write(f'{"EEG + covariates":<30}{auc_both:>7.3f}{bacc_both:>9.3f}\n')
        fh.write(f'{"EEG | covariates (resid)":<30}{auc_res:>7.3f}{bacc_res:>9.3f}\n\n')
        fh.write(f'confound-controlled permutation p = {p_res:.4f}  ({sig_str})  '
                 f'({N_PERM} perms, null AUC mean = {null_mean:.3f})\n\n')
        fh.write(f'SENSITIVITY — subjects with real cycle-phase data '
                 f'(n={n_phase}, no phase imputation)\n')
        fh.write(f'  EEG only {auc_eeg_r:.3f} | covariates only {auc_cov_r:.3f} | '
                 f'EEG|cov {auc_res_r:.3f}  (perm p={p_res_r:.4f})\n\n')
        fh.write(topo_note + '\n')
    print(f'  saved {os.path.basename(txt_path)}')

    # ── Accumulate summary row ────────────────────────────────────────────────
    summary_rows.append({
        'metric':     metric,
        'window':     WINDOW_SIZE,
        'n':          n_sub,
        'n_HC':       n_hc,
        'n_feats':    n_feats,
        'nan_pct':    round(nan_pct, 1),
        'AUC_cov':    round(auc_cov,  3),
        'AUC_eeg':    round(auc_eeg,  3),
        'AUC_both':   round(auc_both, 3),
        'AUC_res':    round(auc_res,  3),
        'perm_p':     round(p_res,    4),
        'sig':        sig_str,
        'AUC_eeg_restricted': round(auc_eeg_r, 3),
        'AUC_res_restricted': round(auc_res_r, 3),
        'perm_p_restricted':  round(p_res_r,   4),
        'n_restricted': n_phase,
    })


# ============================================================
# SAVE SUMMARY — mirrors summary_{window}.csv / .txt from
# neural_timescales_modeling.ipynb, but for the aligned analysis
# ============================================================
df_summary = pd.DataFrame(summary_rows)
csv_path = os.path.join(OUT_DIR, f'summary_timescales_{WINDOW_SIZE}.csv')
df_summary.to_csv(csv_path, index=False)
print(f'\nSaved summary -> {csv_path}')

txt_path = os.path.join(OUT_DIR, f'summary_timescales_{WINDOW_SIZE}.txt')
with open(txt_path, 'w') as fh:
    fh.write(f'Neural Timescales HC Decoding — Summary ({WINDOW_SIZE} window)\n')
    fh.write('Aligned to classification.ipynb (aperiodic exponent + offset decode)\n')
    fh.write('=' * 70 + '\n')
    fh.write(f'Model: L2 logistic, LOOCV, C={C_REG}, {N_PERM} permutations\n')
    fh.write(f'Covariates: {", ".join(COV_LABELS)}\n')
    fh.write('Cycle phase: mean-imputed (no missingness indicator)\n')
    fh.write('Target: current HC (group=1) vs non-user (group=2 or 3)\n')
    fh.write('Feature matrix: each metric × EC+EO × ~64ch (same structure as\n')
    fh.write('  exponent+offset × EC+EO × 64ch in classification.ipynb)\n\n')
    fh.write(f'{"metric":<16}{"n":>5}{"AUC_cov":>9}{"AUC_eeg":>9}'
             f'{"AUC_res":>9}{"perm_p":>9}{"sig":>6}{"NaN%":>6}\n')
    fh.write('-' * 69 + '\n')
    for _, r in df_summary.iterrows():
        fh.write(f'{r.metric:<16}{r.n:>5}{r.AUC_cov:>9.3f}{r.AUC_eeg:>9.3f}'
                 f'{r.AUC_res:>9.3f}{r.perm_p:>9.4f}{r.sig:>6}{r.nan_pct:>6.1f}%\n')
    fh.write('\nAUC_cov = covariates only | AUC_eeg = EEG features only\n')
    fh.write('AUC_res = EEG residualized on covariates (confound-controlled)\n')
    fh.write('perm_p  = label-permutation p-value on the residualized model\n')
    fh.write('sig: *** p<0.001  ** p<0.01  * p<0.05  + p<0.10  n.s. = not significant\n\n')
    fh.write('Phase-restricted sensitivity (subjects with observed cycle phase):\n')
    fh.write('-' * 60 + '\n')
    fh.write(f'{"metric":<16}{"n_phase":>8}{"AUC_eeg_r":>10}{"AUC_res_r":>10}'
             f'{"perm_p_r":>10}\n')
    fh.write('-' * 60 + '\n')
    for _, r in df_summary.iterrows():
        fh.write(f'{r.metric:<16}{r.n_restricted:>8}{r.AUC_eeg_restricted:>10.3f}'
                 f'{r.AUC_res_restricted:>10.3f}{r.perm_p_restricted:>10.4f}\n')
    fh.write('\nAperiodic reference (classification.ipynb, full sample, same covariates):\n')
    fh.write('  exponent+offset, EC+EO, 256 features:\n')
    fh.write('  AUC_cov ≈ 0.51 | AUC_eeg ≈ 0.70 | AUC_res ≈ 0.71 | perm p ≈ 0.02\n')
    fh.write('  (restricted n=50: AUC_eeg ≈ 0.70, AUC_res ≈ 0.71)\n')

print(f'Saved summary -> {txt_path}')
print(f'\nAll outputs in: {OUT_DIR}')
