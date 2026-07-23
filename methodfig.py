"""
Method figures from real sub-01 EEG, matching the study pipeline:
  bandpass 1-40, resample 256, Welch PSD (2 s), specparam fixed (offset+exponent), fit 1-40.
  21_raw_to_psd.png            raw timeseries -> PSD
  22_specparam_decomposition.png   PSD -> aperiodic fit (offset, exponent) + peaks
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.signal import welch
import mne; mne.set_log_level("ERROR")
from specparam import SpectralModel

BIDS="/sessions/vigilant-cool-goodall/mnt/ds007615/ds007615"
OUT=f"{BIDS}/derivatives/preproc/figures"
PINK="#FF66C4"; LIGHT="#FFC7E9"; DARK="#B34789"; INK="#3A2233"; MUTE="#8A6E7C"
plt.rcParams.update({"savefig.dpi":160,"axes.titleweight":"bold","axes.titlecolor":DARK,
    "axes.edgecolor":DARK,"axes.labelcolor":INK,"font.size":13,
    "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":1.2})

# ---- load + preprocess one posterior channel (eyes-closed -> alpha) ----
raw=mne.io.read_raw_brainvision(f"{BIDS}/sub-01/eeg/sub-01_task-rest_acq-ec_eeg.vhdr",preload=True)
raw.pick(["Oz"]); raw.filter(1.,40.); raw.resample(256)
sf=raw.info["sfreq"]; x=raw.get_data()[0]*1e6            # microvolts
# Welch PSD (2 s segments)
nper=int(2*sf)
freqs,psd=welch(x,fs=sf,nperseg=nper,noverlap=nper//2)
sel=(freqs>=1)&(freqs<=45); f,P=freqs[sel],psd[sel]

# ---- specparam fit (fixed: offset + exponent) ----
sm=SpectralModel(peak_width_limits=(1,8),max_n_peaks=6,aperiodic_mode="fixed",verbose=False)
sm.fit(f,P,freq_range=(1,40))
offset,exponent=sm.get_params("aperiodic")    # [offset, exponent]  (specparam 2.0)
ff=sm.data.freqs
data_log=sm.data.power_spectrum               # log10 power (data, fit range)
full_log=sm.results.model.modeled_spectrum    # log10 power (full model)
ap_log=offset-exponent*np.log10(ff)           # aperiodic component (fixed)
r2=float(np.corrcoef(data_log,full_log)[0,1]**2)
print(f"Oz  exponent={exponent:.2f}  offset={offset:.2f}  R^2={r2:.3f}")

# ============================================================
# FIGURE 21 — raw EEG -> PSD
# ============================================================
fig=plt.figure(figsize=(13,4.6))
gL=fig.add_axes([0.05,0.17,0.40,0.70]); gR=fig.add_axes([0.62,0.17,0.35,0.70])
t0=120.0; win=4.0; i0=int(t0*sf); i1=int((t0+win)*sf)
tt=np.arange(i1-i0)/sf
gL.plot(tt,x[i0:i1],color=DARK,lw=1.1)
gL.set_title("Raw EEG  (channel Oz, eyes-closed)")
gL.set_xlabel("time (s)"); gL.set_ylabel("amplitude (µV)"); gL.set_xlim(0,win)
gL.margins(y=0.15)
# PSD (log-log)
gR.semilogy(f,P,color=DARK,lw=2)
gR.set_title("Power spectral density  (Welch)")
gR.set_xlabel("frequency (Hz)"); gR.set_ylabel("power (µV²/Hz)")
gR.set_xlim(1,45)
gR.axvspan(8,13,color=LIGHT,alpha=.55,lw=0); gR.text(10.5,gR.get_ylim()[1]*0.35,"α",ha="center",color=DARK,fontsize=16,fontweight="bold")
# arrow between
arr=FancyArrowPatch((0.475,0.52),(0.565,0.52),transform=fig.transFigure,
    arrowstyle="-|>",mutation_scale=32,lw=3,color=PINK)
fig.add_artist(arr)
fig.text(0.52,0.60,"Welch PSD",ha="center",va="center",color=DARK,fontsize=11,fontweight="bold")
fig.suptitle("From raw signal to power spectrum",fontsize=18,fontweight="bold",color=DARK,x=0.52,y=1.02)
fig.savefig(f"{OUT}/21_raw_to_psd.png",bbox_inches="tight",facecolor="white"); plt.close(fig)
print("saved 21_raw_to_psd.png")

# ============================================================
# FIGURE 22 — PSD -> specparam aperiodic decomposition
# ============================================================
fig,ax=plt.subplots(figsize=(9.5,6.2))
ax.plot(ff,data_log,color=DARK,lw=2.4,label="observed spectrum",zorder=4)
ax.plot(ff,full_log,color=PINK,lw=2.2,ls="-",label="full model fit",zorder=5)
ax.plot(ff,ap_log,color=PINK,lw=2.4,ls="--",label="aperiodic fit  (1/f)",zorder=3)
ax.fill_between(ff,ap_log,np.maximum(data_log,ap_log),color=LIGHT,alpha=.6,label="periodic (peaks)",zorder=2)
ax.set_xscale("log")
ax.set_xlim(1,40); ax.set_xticks([1,2,5,10,20,40]); ax.set_xticklabels([1,2,5,10,20,40])
ax.set_xlabel("frequency (Hz, log scale)"); ax.set_ylabel("log₁₀ power")
ax.set_ylim(top=data_log.max()+0.80)
ax.set_title("Deriving the aperiodic parameters from the PSD", pad=16)

# offset annotation (intercept at 1 Hz)
y0=offset
ax.scatter([1],[y0],s=90,color=PINK,zorder=6,edgecolor=DARK)
ax.annotate("offset\n(vertical position)",xy=(1,y0),xytext=(1.5,y0+0.30),
            color=DARK,fontsize=12,fontweight="bold",
            arrowprops=dict(arrowstyle="-|>",color=DARK,lw=1.6))
# exponent = slope: draw a slope triangle on the aperiodic line
xa,xb=4.0,12.0
ya=offset-exponent*np.log10(xa); yb=offset-exponent*np.log10(xb)
ax.plot([xa,xb],[ya,ya],color=MUTE,lw=1.6)
ax.plot([xb,xb],[ya,yb],color=MUTE,lw=1.6)
ax.annotate(f"exponent χ = {exponent:.2f}\n(steepness of 1/f slope)",
            xy=(xb,(ya+yb)/2),xytext=(13.5,ya+0.15),color=DARK,fontsize=12,fontweight="bold",
            arrowprops=dict(arrowstyle="-|>",color=DARK,lw=1.6))
# peak label (to the right of the alpha peak, clear of the title)
ax.annotate("periodic peaks\n(e.g. alpha)",xy=(11.5,np.interp(11,ff,data_log)),
            xytext=(17,data_log.max()+0.06),color=DARK,fontsize=12,fontweight="bold",ha="center",
            arrowprops=dict(arrowstyle="-|>",color=DARK,lw=1.6))
ax.text(0.985,0.03,f"specparam · fixed mode · fit 1–40 Hz · R² = {r2:.2f}",
        transform=ax.transAxes,ha="right",va="bottom",fontsize=10,color=MUTE)
ax.legend(loc="lower left",fontsize=11,framealpha=.92)
fig.savefig(f"{OUT}/22_specparam_decomposition.png",bbox_inches="tight",facecolor="white"); plt.close(fig)
print("saved 22_specparam_decomposition.png")
