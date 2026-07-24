"""
Neural Timescales — High-Definition Presentation Figures
Aligned to figures_HD.ipynb (aperiodic parameter figures)

═══════════════════════════════════════════════════════════════════════════════
FIGURE CATALOG  (continues from figures_HD.ipynb, which ends at fig 24)
───────────────────────────────────────────────────────────────────────────────
Each timescale metric gets its OWN independent figure — NO multi-metric grids.
This matches the project decision: choose 1–2 metrics to present, compare
all 4 here and then select.

Figures for KNEE_FREQ_HZ (aperiodic method timescale):
  25_knee_freq_by_hc.png            violin by HC group  [≡ fig 14]
  26_knee_freq_decode_model.png      final decode model bar chart  [≡ fig 17]
  27_knee_freq_coef_topomap.png      logistic coef topomap  [≡ hc_coef_topomap.png]
  28_knee_freq_by_formulation.png    progestin-only vs combined  [≡ fig 18]
  29_knee_freq_predictors.png        standardized β forest  [≡ fig 19]
  30_knee_freq_formulation_topo.png  per-channel FDR t-map  [≡ fig 20]

Figures for ACW_0_MS:
  31_ACW_0_by_hc.png
  32_ACW_0_decode_model.png
  33_ACW_0_coef_topomap.png
  34_ACW_0_by_formulation.png
  35_ACW_0_predictors.png
  36_ACW_0_formulation_topo.png

Figures for ACW_50_MS:
  37_ACW_50_by_hc.png
  38_ACW_50_decode_model.png
  39_ACW_50_coef_topomap.png
  40_ACW_50_by_formulation.png
  41_ACW_50_predictors.png
  42_ACW_50_formulation_topo.png

Figures for ACW_1E_MS:
  43_ACW_1e_by_hc.png
  44_ACW_1e_decode_model.png
  45_ACW_1e_coef_topomap.png
  46_ACW_1e_by_formulation.png
  47_ACW_1e_predictors.png
  48_ACW_1e_formulation_topo.png

MECHANISM FIGURES (one per metric, cycle phase — mirrors mechanism.py):
  53_knee_freq_phase_cyclers.png     Follicular/Ovulatory/Luteal violin (naturally-cycling)
  54_ACW_0_phase_cyclers.png
  55_ACW_50_phase_cyclers.png
  56_ACW_1e_phase_cyclers.png

  57_knee_freq_phase_hc_neg.png      Phase violin in HC users (negative control)
  58_ACW_0_phase_hc_neg.png
  59_ACW_50_phase_hc_neg.png
  60_ACW_1e_phase_hc_neg.png

ADDITIONAL (timescale-specific methods + comparison helper):
  49_knee_freq_vs_age.png            timescale vs age scatter  [≡ fig 13]
  50_ACW_0_vs_age.png
  51_ACW_50_vs_age.png
  52_ACW_1e_vs_age.png
═══════════════════════════════════════════════════════════════════════════════

COLOR SCHEME — exactly matches figures_HD.ipynb:
  PINK  = '#FF66C4'  (primary / current HC / positive)
  LIGHT = '#FFC7E9'  (secondary / past HC)
  DARK  = '#B34789'  (accent / never-user / negative / axis)
  WHITE = '#FFFFFF'  (background)
  GROUP_PAL: Current HC → PINK, Past HC → LIGHT, Never → DARK

INPUTS:
  derivatives/preproc/specparam/knee_frequencies.csv         (2s window)
  derivatives/preproc/specparam/acf_timescales.csv           (2s window)
  derivatives/preproc/specparam/timescales_subject_means_with_meta.csv
  derivatives/models/models_timescales_2s/{metric}_model.pkl
  ds007615-download/participants.tsv
  ds007615-download/phenotype/{lifestyle,bdi,hc_usage}.tsv

OUTPUTS → derivatives/preproc/figures_HD/  (same folder as aperiodic figures)

To use 4s window: set WINDOW_SIZE = '4s'; pkl files load from models_timescales_4s/.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import mne

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG  ← edit here
# ============================================================
WINDOW_SIZE = '4s'   # '2s' (aligned to aperiodic) or '4s'

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
BIDS_ROOT   = os.path.join(BASE_DIR, 'ds007615-download')
DERIV_ROOT  = os.path.join(BASE_DIR, 'derivatives')

if WINDOW_SIZE == '2s':
    SPECPARAM_DIR = os.path.join(DERIV_ROOT, 'preproc', 'specparam')
    KNEE_CSV      = os.path.join(SPECPARAM_DIR, 'knee_frequencies.csv')
    ACF_CSV       = os.path.join(SPECPARAM_DIR, 'acf_timescales.csv')
    MODEL_DIR     = os.path.join(DERIV_ROOT, 'models', 'models_timescales_2s')
elif WINDOW_SIZE == '4s':
    SPECPARAM_DIR = os.path.join(DERIV_ROOT, 'preproc_long', 'specparam')
    KNEE_CSV      = os.path.join(SPECPARAM_DIR, 'knee_frequencies_master_new.csv')
    ACF_CSV       = os.path.join(SPECPARAM_DIR, 'acf_timescales_new.csv')
    MODEL_DIR     = os.path.join(DERIV_ROOT, 'models', 'models_timescales_4s')
else:
    raise ValueError("WINDOW_SIZE must be '2s' or '4s'")

OUT_DIR = os.path.join(DERIV_ROOT, 'preproc', 'figures_HD')
os.makedirs(OUT_DIR, exist_ok=True)

PARTICIPANTS  = os.path.join(BIDS_ROOT, 'participants.tsv')
PHENOTYPE_DIR = os.path.join(BIDS_ROOT, 'phenotype')

# ── Brand palette (identical to figures_HD.ipynb) ──────────────────────────
PINK  = '#FF66C4'
LIGHT = '#FFC7E9'
DARK  = '#B34789'
WHITE = '#FFFFFF'
INK   = '#3A2233'
MUTE  = '#8A6E7C'
GROUP_PAL  = {'Current HC': PINK, 'Past HC': LIGHT, 'Never': DARK}
GROUP_ORDER = ['Current HC', 'Past HC', 'Never']
CMAP_DIV = LinearSegmentedColormap.from_list('pink_div', [DARK, WHITE, PINK])
CMAP_SEQ = LinearSegmentedColormap.from_list('pink_seq', [WHITE, LIGHT, PINK, DARK])

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams.update({
    'figure.dpi':       120,
    'savefig.dpi':      300,
    'axes.titleweight': 'bold',
    'axes.titlecolor':  DARK,
    'axes.edgecolor':   DARK,
    'axes.labelcolor':  '#333333',
    'axes.spines.top':  False,
    'axes.spines.right': False,
    'grid.color':       '#F3DEEA',
    'font.size':        13,
})

# ── Metric metadata ─────────────────────────────────────────────────────────
# Each entry: (col_name, figure_prefix, y_label, unit_label, method_label)
METRICS_META = [
    ('knee_freq_hz', '25_knee_freq',
     'Knee frequency (Hz)',       'Hz',  'Knee frequency (specparam knee mode)'),
    ('ACW_0_ms',     '31_ACW_0',
     'ACW-0 timescale (ms)',      'ms',  'ACW-0 (autocorrelation zero-crossing)'),
    ('ACW_50_ms',    '37_ACW_50',
     'ACW-50 timescale (ms)',     'ms',  'ACW-50 (50% ACF decay)'),
    ('ACW_1e_ms',    '43_ACW_1e',
     'ACW-1/e timescale (ms)',    'ms',  'ACW-1/e (exponential decay constant)'),
]

# For vs-age figures (additional section, figs 49-52)
AGE_FIG_NUMS = {'knee_freq_hz': 49, 'ACW_0_ms': 50, 'ACW_50_ms': 51, 'ACW_1e_ms': 52}

# For cycle-phase mechanism figures (figs 53-60)
PHASE_CYCLERS_NUMS = {'knee_freq_hz': 53, 'ACW_0_ms': 54, 'ACW_50_ms': 55, 'ACW_1e_ms': 56}
PHASE_HC_NEG_NUMS  = {'knee_freq_hz': 57, 'ACW_0_ms': 58, 'ACW_50_ms': 59, 'ACW_1e_ms': 60}

PHASE_ORDER = ['Follicular', 'Ovulatory', 'Luteal']
PHASE_PAL   = {'Follicular': LIGHT, 'Ovulatory': PINK, 'Luteal': DARK}


# ============================================================
# LOAD DATA
# ============================================================
def _load_tsv(path, int_subject=True):
    d = pd.read_csv(path, sep='\t', na_values='n/a')
    d['subject'] = d['participant_id'].str.replace('sub-', '', regex=False)
    if int_subject:
        d['subject'] = d['subject'].astype(int)
    return d.set_index('subject')

parts = _load_tsv(PARTICIPANTS)
life  = _load_tsv(os.path.join(PHENOTYPE_DIR, 'lifestyle.tsv'))
bdi   = _load_tsv(os.path.join(PHENOTYPE_DIR, 'bdi.tsv'))
hc    = _load_tsv(os.path.join(PHENOTYPE_DIR, 'hc_usage.tsv'))

# Per-channel raw data (for FDR topomaps)
df_knee = pd.read_csv(KNEE_CSV, dtype={'subject': int})
# Quality filter for knee fits (mirrors classification.py + mechanisms.py)
R2_THRESH = 0.95
if 'r_squared' in df_knee.columns:
    df_knee.loc[df_knee['r_squared'] < R2_THRESH, 'knee_freq_hz'] = np.nan
df_acf  = pd.read_csv(ACF_CSV,  dtype={'subject': int})

# ── Subject-level means — always built from raw per-channel CSVs ─────────────
# NOTE: timescales_subject_means_with_meta.csv stores the knee timescale as
# 'tau_knee_ms' (ms), not 'knee_freq_hz' (Hz).  Building from df_knee and
# df_acf directly guarantees the column names match METRICS_META throughout.
k_subj = df_knee.groupby('subject')[['knee_freq_hz']].mean().reset_index()
a_subj = df_acf.groupby('subject')[['ACW_0_ms', 'ACW_50_ms', 'ACW_1e_ms']].mean().reset_index()
df_subj = k_subj.merge(a_subj, on='subject', how='outer')
# Attach phenotype metadata
df_subj['group']          = parts.loc[df_subj.subject.values, 'group'].values
df_subj['age']            = parts.loc[df_subj.subject.values, 'age'].values
df_subj['edu']            = parts.loc[df_subj.subject.values, 'edu'].values
df_subj['main_prev']      = hc.loc[df_subj.subject.values, 'main_prev'].values
df_subj['medication_use'] = life.loc[df_subj.subject.values, 'medication_use'].values
df_subj['daily_nicotine'] = life.loc[df_subj.subject.values, 'daily_nicotine'].values
df_subj['alcohol_units']  = life.loc[df_subj.subject.values, 'alcohol_units'].values
df_subj['bdi_total']      = bdi.loc[df_subj.subject.values, 'bdi_total'].values

df_subj['HC group']   = df_subj['group'].map({1: 'Current HC', 2: 'Past HC', 3: 'Never'})
df_subj['mens_phase'] = hc.loc[df_subj.subject.values, 'mens_phase'].values
df_subj['phase']      = df_subj['mens_phase'].map(
    {1: 'Follicular', 2: 'Ovulatory', 3: 'Luteal'})
present_groups = [g for g in GROUP_ORDER if g in df_subj['HC group'].values]


# ── MNE info for topomaps ───────────────────────────────────────────────────
channels_all = sorted(df_acf['channel'].unique())
mne_info = mne.create_info(channels_all, sfreq=1.0, ch_types='eeg')
mne_info.set_montage(mne.channels.make_standard_montage('standard_1005'), on_missing='ignore')


# ============================================================
# HELPER: save figure
# ============================================================
def save(fig, fname):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  saved {fname}')


# ============================================================
# MAIN LOOP — one full figure set per metric
# ============================================================
for col, prefix, ylabel, unit, method_label in METRICS_META:

    # ── Source dataframe for per-channel data ─────────────────────────────────
    df_ch = df_knee if col == 'knee_freq_hz' else df_acf

    print(f'\n── {col} ─────────────────────────────────────────────')

    # ── Load decode results from pkl (produced by neural_timescales_classification.py)
    pkl_path = os.path.join(MODEL_DIR, f'{col}_model.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as fh:
            mdl = pickle.load(fh)
        auc_cov = None   # read from text file
        auc_eeg = mdl.get('auc_eeg')
        auc_res = mdl.get('auc_res')
        p_res   = mdl.get('p_res')
        coef_by_feat = mdl.get('coef_by_feat', {})
    else:
        print(f'  [WARNING] No pkl found at {pkl_path}. '
              f'Run neural_timescales_classification.py first.')
        auc_eeg = auc_res = p_res = None
        coef_by_feat = {}

    # Parse covariates-only AUC from the text results file
    txt_path = os.path.join(MODEL_DIR, f'results_{col}.txt')
    auc_cov_val = 0.50   # fallback
    if os.path.exists(txt_path):
        with open(txt_path) as fh:
            for line in fh:
                if 'covariates only' in line:
                    try:
                        auc_cov_val = float(line.split()[2])
                    except Exception:
                        pass
                    break

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE A: Timescale by HC group — violin  [≡ fig 14 exponent_by_group]
    # ──────────────────────────────────────────────────────────────────────────
    sub_a = df_subj[['subject', 'HC group', col]].dropna()
    g_groups = {g: sub_a[sub_a['HC group'] == g][col].dropna().values
                for g in present_groups}

    # ANOVA across 3 groups
    try:
        F_anova, p_anova = stats.f_oneway(*[v for v in g_groups.values() if len(v) >= 3])
    except Exception:
        F_anova, p_anova = np.nan, np.nan

    # Current HC vs non-current t-test
    cur_v = sub_a[sub_a['HC group'] == 'Current HC'][col].values
    non_v = sub_a[sub_a['HC group'] != 'Current HC'][col].values
    if len(cur_v) >= 3 and len(non_v) >= 3:
        t_hc, p_hc = stats.ttest_ind(cur_v, non_v)
        d_hc = (cur_v.mean() - non_v.mean()) / np.sqrt(
            ((len(cur_v)-1)*cur_v.var(ddof=1) + (len(non_v)-1)*non_v.var(ddof=1)) /
            (len(cur_v) + len(non_v) - 2))
    else:
        t_hc = p_hc = d_hc = np.nan

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sns.violinplot(data=sub_a, x='HC group', y=col, order=present_groups,
                   hue='HC group', palette=GROUP_PAL, legend=False,
                   inner='quartile', cut=0, ax=ax)
    sns.stripplot(data=sub_a, x='HC group', y=col, order=present_groups,
                  color=DARK, size=3.5, alpha=0.55, ax=ax)
    ax.set_title(f'{method_label}\nby HC status')
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    ax.text(0.5, -0.22,
            f'Current HC vs non-current: d = {d_hc:+.2f}, p = {p_hc:.3f}  ·  '
            f'3-group ANOVA p = {p_anova:.3f}',
            transform=ax.transAxes, ha='center', va='top', fontsize=10.5, color=DARK)
    save(fig, f'{prefix}_by_hc.png')

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE B: Final decode model bar chart  [≡ fig 17]
    # ──────────────────────────────────────────────────────────────────────────
    if auc_eeg is not None and auc_res is not None:
        p_str   = f'{p_res:.3f}'
        sig_str = ('***' if p_res < 0.001 else '**' if p_res < 0.01
                   else '*' if p_res < 0.05 else '+' if p_res < 0.10 else 'n.s.')

        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        bars   = ['Confounds\nonly', 'EEG\nonly', 'EEG | confounds\n(residualized)']
        vals   = [auc_cov_val, auc_eeg, auc_res]
        colors = [LIGHT, PINK, DARK]
        b = ax.bar(bars, vals, color=colors, edgecolor=DARK, width=0.62)
        ax.axhline(0.5, color='0.5', ls='--', lw=1.6, zorder=0)
        ax.text(-0.45, 0.505, 'chance', color='0.4', fontsize=10, ha='left', va='bottom')
        for rect, v in zip(b, vals):
            ax.annotate(f'{v:.2f}',
                        (rect.get_x() + rect.get_width()/2, v),
                        ha='center', va='bottom', fontsize=14,
                        color=DARK, fontweight='bold')
        if p_res is not None:
            ax.annotate(f'permutation p = {p_str}  ({sig_str})',
                        (2, auc_res), xytext=(2, auc_res + 0.06),
                        ha='center', fontsize=12, color=DARK, fontweight='bold')
        ax.set_ylim(0.4, min(0.85, max(vals) + 0.15))
        ax.set_ylabel('LOSO AUC')
        ax.set_title(f'HC use decodes from {method_label}\nafter full confound control')
        ax.text(0.5, -0.16,
                'Confounds = age · nicotine · alcohol · education · BDI · cycle phase',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=10.5, color='0.4')
        save(fig, f'{prefix}_decode_model.png')
    else:
        print(f'  [SKIP] decode model figure — no pkl found')

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE C: Logistic coefficient topomap  [≡ hc_coef_topomap.png]
    # ──────────────────────────────────────────────────────────────────────────
    if coef_by_feat:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
        for ax, acq in zip(axes, ['ec', 'eo']):
            vals = np.array([coef_by_feat.get(f'{col}_{acq}_{ch}', np.nan)
                             for ch in channels_all])
            vmax = np.nanmax(np.abs(vals)) or 1.0
            try:
                im, _ = mne.viz.plot_topomap(
                    vals, mne_info, axes=ax, show=False,
                    cmap=CMAP_DIV, vlim=(-vmax, vmax), contours=0
                )
                ax.set_title(f'{"Eyes-closed" if acq == "ec" else "Eyes-open"}',
                             fontsize=12, fontweight='bold', color=DARK)
            except Exception as e:
                ax.text(0.5, 0.5, str(e), transform=ax.transAxes,
                        ha='center', va='center', fontsize=9)
        fig.suptitle(f'Logistic regression coefficients: {method_label}\n'
                     f'Pink = predicts current HC use  |  EEG-only model',
                     fontsize=11, color=DARK, fontweight='bold')
        save(fig, f'{prefix}_coef_topomap.png')
    else:
        print(f'  [SKIP] coef topomap — no pkl found')

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE D: Timescale by HC formulation  [≡ fig 18 — exponent_by_formulation]
    # HC users (group=1) only: progestin-only (main_prev=1) vs combined (main_prev=2)
    # ──────────────────────────────────────────────────────────────────────────
    hcu = df_subj[(df_subj['group'] == 1) & df_subj['main_prev'].notna()].copy()
    hcu['Formulation'] = hcu['main_prev'].map({1: 'Progestin-only', 2: 'Combined\n(+ estrogen)'})
    hcu_form = hcu[['Formulation', col]].dropna()
    form_order = ['Progestin-only', 'Combined\n(+ estrogen)']

    prog_v = hcu_form[hcu_form.Formulation == 'Progestin-only'][col].values
    comb_v = hcu_form[hcu_form.Formulation == 'Combined\n(+ estrogen)'][col].values

    if len(prog_v) >= 3 and len(comb_v) >= 3:
        t_form, p_form = stats.ttest_ind(prog_v, comb_v)
        d_form = (comb_v.mean() - prog_v.mean()) / np.sqrt(
            ((len(comb_v)-1)*comb_v.var(ddof=1) + (len(prog_v)-1)*prog_v.var(ddof=1)) /
            (len(comb_v) + len(prog_v) - 2))
    else:
        t_form = p_form = d_form = np.nan

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sns.boxplot(data=hcu_form, x='Formulation', y=col,
                order=form_order, palette={'Progestin-only': LIGHT,
                                           'Combined\n(+ estrogen)': PINK},
                width=0.55, fliersize=0, ax=ax, linecolor=DARK)
    sns.stripplot(data=hcu_form, x='Formulation', y=col,
                  order=form_order, color=DARK, size=5, alpha=0.65, ax=ax)
    ax.set_title(f'{method_label}\nby contraceptive formulation  (HC users only)')
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    note = (f'Combined vs progestin-only: d = {d_form:+.2f}, p = {p_form:.3f}  '
            f'(n_prog={len(prog_v)}, n_comb={len(comb_v)})')
    ax.text(0.5, -0.20, note, transform=ax.transAxes,
            ha='center', va='top', fontsize=10.5, color=DARK)
    save(fig, f'{prefix}_by_formulation.png')

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE E: Standardized predictor forest  [≡ fig 19 — exponent_predictors]
    # OLS: timescale ~ HC + age_z + med + nic + alc_z
    # ──────────────────────────────────────────────────────────────────────────
    dd = df_subj[[col, 'group', 'age', 'medication_use',
                  'daily_nicotine', 'alcohol_units', 'bdi_total']].copy()
    dd['HC']    = (dd['group'] == 1).astype(int)
    dd['med']   = (dd['medication_use'] == 1).astype(int)
    dd['nic']   = (dd['daily_nicotine'] == 1).astype(int)
    dd['alc']   = dd['alcohol_units'].astype(float)
    dd['bdi_z'] = (dd['bdi_total'] - dd['bdi_total'].mean()) / dd['bdi_total'].std()
    # z-score continuous predictors
    for src in ['age', 'alc']:
        s = dd[src].astype(float)
        dd[f'{src}_z'] = (s - s.mean()) / s.std()
    dd['y_z'] = (dd[col] - dd[col].mean()) / dd[col].std()
    dd = dd.dropna(subset=['y_z', 'HC', 'age_z', 'med', 'nic', 'alc_z'])

    if len(dd) >= 10:
        m_ols  = smf.ols('y_z ~ HC + age_z + med + nic + alc_z', data=dd).fit()
        ci_ols = m_ols.conf_int()
        term_map = {'HC': 'HC use (current)', 'age_z': 'Age',
                    'med': 'Medication use', 'nic': 'Daily nicotine', 'alc_z': 'Alcohol'}
        term_order = ['Daily nicotine', 'Age', 'HC use (current)', 'Medication use', 'Alcohol']
        rows_f = []
        for t, lbl in term_map.items():
            rows_f.append((lbl, m_ols.params[t], ci_ols.loc[t, 0],
                           ci_ols.loc[t, 1], m_ols.pvalues[t]))
        Rf = pd.DataFrame(rows_f, columns=['term', 'beta', 'lo', 'hi', 'p'])
        Rf = Rf.set_index('term').reindex(term_order)

        fig, ax = plt.subplots(figsize=(8.6, 5.6))
        ax.axvline(0, color=MUTE, lw=1.4, ls='--')
        yy = np.arange(len(term_order))[::-1]
        ax.errorbar(Rf.beta, yy,
                    xerr=[Rf.beta - Rf.lo, Rf.hi - Rf.beta],
                    fmt='o', color=PINK, ecolor=PINK,
                    elinewidth=2, capsize=3, ms=9, mec=DARK)
        for y_pos, (_, row) in zip(yy, Rf.iterrows()):
            if row.p < 0.05:
                ax.text(row.hi + 0.02, y_pos, '*', color=DARK,
                        fontsize=16, va='center', fontweight='bold')
        ax.set_yticks(yy)
        ax.set_yticklabels(term_order)
        ax.set_xlabel('standardised β  (effect on timescale)')
        ax.set_title(f'What predicts {method_label}?')
        ax.text(0.5, -0.17,
                f'* p < .05   ·   n = {len(dd)}   ·   '
                f'model R² = {m_ols.rsquared:.3f}',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=10.5, color=MUTE)
        save(fig, f'{prefix}_predictors.png')
    else:
        print(f'  [SKIP] predictor forest — too few complete cases')

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE F: Per-channel FDR topomap — formulation effect  [≡ fig 20]
    # t-values (combined − progestin-only), age-adjusted, FDR BH
    # ──────────────────────────────────────────────────────────────────────────
    if col in df_ch.columns and 'main_prev' in hc.columns:
        hc_subs = parts.index[parts['group'] == 1]
        df_hcu_ch = df_ch[df_ch['subject'].isin(hc_subs)].copy()
        df_hcu_ch['combined'] = hc.loc[df_hcu_ch['subject'].values, 'main_prev'].apply(
            lambda x: (1.0 if x == 2 else 0.0) if not pd.isna(x) else np.nan).values
        df_hcu_ch['age_cov'] = parts.loc[df_hcu_ch['subject'].values, 'age'].values

        # Use average across EC+EO per subject per channel
        df_avg = df_hcu_ch.groupby(['subject', 'channel'])[
            [col, 'combined', 'age_cov']].mean().reset_index()

        tvals = np.full(len(channels_all), np.nan)
        pvals = np.full(len(channels_all), np.nan)
        for i, ch in enumerate(channels_all):
            sub_ch = df_avg[df_avg.channel == ch][['combined', 'age_cov', col]].dropna()
            if sub_ch['combined'].nunique() < 2 or len(sub_ch) < 6:
                continue
            sub_ch.columns = ['combined', 'age', 'y']
            try:
                m_ch = smf.ols('y ~ combined + age', data=sub_ch).fit()
                tvals[i] = m_ch.tvalues['combined']
                pvals[i] = m_ch.pvalues['combined']
            except Exception:
                pass

        ok = ~np.isnan(pvals)
        rej = np.zeros(len(channels_all), bool)
        if ok.sum() >= 5:
            rej_ok, _, _, _ = multipletests(pvals[ok], alpha=0.05, method='fdr_bh')
            rej[ok] = rej_ok

        n_sig = int(rej.sum())
        vmax  = np.nanmax(np.abs(tvals))

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        mask_params = dict(marker='o', markerfacecolor=INK, markeredgecolor=INK, markersize=7)
        try:
            im, _ = mne.viz.plot_topomap(
                np.nan_to_num(tvals), mne_info, axes=ax, show=False,
                cmap=CMAP_DIV, vlim=(-vmax, vmax), contours=0,
                mask=rej, mask_params=mask_params, sensors=True
            )
            fig.colorbar(im, ax=ax, shrink=0.7,
                         label='t  (combined − progestin-only)')
        except Exception as e:
            ax.text(0.5, 0.5, str(e), transform=ax.transAxes, ha='center', va='center')
        ax.set_title(f'{method_label}\nformulation effect per channel\n'
                     f'({n_sig} channels FDR q < .05)',
                     fontsize=11, fontweight='bold', color=DARK)
        ax.text(0.5, -0.08,
                'Dots = FDR-significant (BH q < .05)  ·  age-adjusted',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=10, color=MUTE)
        save(fig, f'{prefix}_formulation_topo.png')
    else:
        print(f'  [SKIP] formulation topomap — column {col} not found')

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE G: Cycle phase violin — naturally-cycling women  [≡ mechanism.py]
    # Groups: Follicular / Ovulatory / Luteal  (non-HC only, observed phase)
    # ──────────────────────────────────────────────────────────────────────────
    nat = df_subj[(df_subj['group'].isin([2, 3])) & df_subj['phase'].notna()].copy()
    nat_col = nat[['phase', col]].dropna()

    if len(nat_col) >= 9:
        grps_g = [nat_col.loc[nat_col['phase'] == p, col].values for p in PHASE_ORDER]
        if all(len(g) >= 3 for g in grps_g):
            F_ph, p_ph = stats.f_oneway(*grps_g)
            # Luteal vs first-half
            lut_v = nat_col.loc[nat_col['phase'] == 'Luteal', col].values
            fh_v  = nat_col.loc[nat_col['phase'] != 'Luteal', col].values
            t_ph, p_lf = stats.ttest_ind(lut_v, fh_v)
            d_ph  = (lut_v.mean() - fh_v.mean()) / np.sqrt(
                ((len(lut_v)-1)*lut_v.var(ddof=1) + (len(fh_v)-1)*fh_v.var(ddof=1)) /
                (len(lut_v) + len(fh_v) - 2)) if (len(lut_v) >= 2 and len(fh_v) >= 2) else np.nan
        else:
            F_ph = p_ph = t_ph = p_lf = d_ph = np.nan

        fig, ax = plt.subplots(figsize=(7.5, 5.6))
        sns.violinplot(data=nat_col, x='phase', y=col, order=PHASE_ORDER,
                       hue='phase', palette=PHASE_PAL, legend=False,
                       inner='quartile', cut=0, ax=ax)
        sns.stripplot(data=nat_col, x='phase', y=col, order=PHASE_ORDER,
                      color=INK, size=4, alpha=0.55, ax=ax)
        ax.set_title(f'{method_label}\nby menstrual phase  (naturally-cycling women, n={len(nat_col)})')
        ax.set_xlabel('Menstrual phase')
        ax.set_ylabel(ylabel)
        stats_note = (f'3-group ANOVA: F={F_ph:.2f}, p={p_ph:.3f}  ·  '
                      f'Luteal vs first-half: d={d_ph:+.2f}, p={p_lf:.3f}')
        ax.text(0.5, -0.20, stats_note,
                transform=ax.transAxes, ha='center', va='top',
                fontsize=10.5, color=DARK)
        fig_num_g = PHASE_CYCLERS_NUMS[col]
        metric_short = col.replace('_ms', '').replace('_', '')
        save(fig, f'{fig_num_g}_{metric_short}_phase_cyclers.png')
    else:
        print(f'  [SKIP] phase cyclers violin — too few naturally-cycling with phase (n={len(nat_col)})')

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE H: Cycle phase violin — HC users (negative control)  [≡ mechanism2.py B]
    # Expected NULL: HC suppresses natural cycle → phase shouldn't matter
    # ──────────────────────────────────────────────────────────────────────────
    hcu_ph = df_subj[(df_subj['group'] == 1) & df_subj['phase'].notna()].copy()
    hcu_col = hcu_ph[['phase', col]].dropna()

    if len(hcu_col) >= 6:
        grps_h = [hcu_col.loc[hcu_col['phase'] == p, col].values for p in PHASE_ORDER]
        if all(len(g) >= 2 for g in grps_h):
            F_hc, p_hc = stats.f_oneway(*grps_h)
        else:
            F_hc, p_hc = np.nan, np.nan

        fig, ax = plt.subplots(figsize=(7.5, 5.6))
        sns.violinplot(data=hcu_col, x='phase', y=col, order=PHASE_ORDER,
                       hue='phase', palette=PHASE_PAL, legend=False,
                       inner='quartile', cut=0, ax=ax)
        sns.stripplot(data=hcu_col, x='phase', y=col, order=PHASE_ORDER,
                      color=INK, size=4, alpha=0.55, ax=ax)
        ax.set_title(f'{method_label}\nby reported phase — HC users only  '
                     f'(negative control, n={len(hcu_col)})')
        ax.set_xlabel('Reported menstrual phase (HC-clamped)')
        ax.set_ylabel(ylabel)
        note_h = (f'3-group ANOVA: F={F_hc:.2f}, p={p_hc:.3f}  '
                  f'·  Expected null (HC suppresses natural cycle)')
        ax.text(0.5, -0.20, note_h,
                transform=ax.transAxes, ha='center', va='top',
                fontsize=10.5, color=MUTE)
        fig_num_h = PHASE_HC_NEG_NUMS[col]
        save(fig, f'{fig_num_h}_{metric_short}_phase_hc_neg.png')
    else:
        print(f'  [SKIP] phase HC negative control — too few HC with phase (n={len(hcu_col)})')

    print(f'  ✓ All {col} figures done')

print('\n')

# ============================================================
# ADDITIONAL FIGURES: Timescale vs age  [≡ fig 13]
# ============================================================
for col, prefix, ylabel, unit, method_label in METRICS_META:
    fig_num = AGE_FIG_NUMS[col]
    sub_age = df_subj[['age', col]].dropna()
    if len(sub_age) < 5:
        continue

    r_val, p_val = stats.pearsonr(sub_age['age'], sub_age[col])

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sns.regplot(data=sub_age, x='age', y=col,
                color=PINK,
                scatter_kws=dict(s=55, alpha=0.75, edgecolor=DARK),
                line_kws=dict(color=DARK), ax=ax)
    ax.set_title(f'{method_label}\nvs age')
    ax.set_xlabel('age (years)')
    ax.set_ylabel(ylabel)
    ax.text(0.03, 0.05, f'r = {r_val:.2f}, p = {p_val:.3f}',
            transform=ax.transAxes, fontsize=12, color=DARK, fontweight='bold')
    save(fig, f'{fig_num}_{col}_vs_age.png')

print(f'\nAll neural timescales HD figures saved to:\n  {OUT_DIR}')
print('\nFigure index summary:')
for col, prefix, ylabel, unit, method_label in METRICS_META:
    base_num = int(prefix.split('_')[0])
    print(f'\n  {method_label}:')
    labels = ['by_hc', 'decode_model', 'coef_topomap', 'by_formulation',
              'predictors', 'formulation_topo']
    for i, lbl in enumerate(labels):
        print(f'    fig {base_num + i}: {prefix}_{lbl}.png')
print('\n  Timescale vs age:')
for col, prefix, ylabel, unit, method_label in METRICS_META:
    print(f'    fig {AGE_FIG_NUMS[col]}: {AGE_FIG_NUMS[col]}_{col}_vs_age.png  [{method_label}]')
