"""
specparam aperiodic-parameter figure: clean channel-averaged real spectrum
(sub-01 eyes-closed), annotating only the derived aperiodic params (offset & exponent).
  -> 22_specparam_decomposition.png
"""
import numpy as np, mne; mne.set_log_level("ERROR")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from specparam import SpectralModel

BIDS="/sessions/vigilant-cool-goodall/mnt/ds007615/ds007615"
OUT=f"{BIDS}/derivatives/preproc/figures"
PINK="#FF66C4"; LIGHT="#FFC7E9"; DARK="#B34789"; INK="#3A2233"; MUTE="#9A6B84"
plt.rcParams.update({"font.size":13,"axes.titleweight":"bold","axes.titlecolor":DARK,
    "axes.edgecolor":DARK,"axes.labelcolor":INK,
    "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":1.2})

# ---- clean spectrum: average Welch PSD across all channels (sub-01 EC) ----
raw=mne.io.read_raw_brainvision(f"{BIDS}/sub-01/eeg/sub-01_task-rest_acq-ec_eeg.vhdr",preload=True)
raw.filter(1.,40.); raw.resample(256)
X=raw.get_data()*1e6
f,Pall=welch(X,fs=256,nperseg=512,noverlap=256,axis=1); P=Pall.mean(0)
sm=SpectralModel(peak_width_limits=(1,8),max_n_peaks=6,aperiodic_mode="fixed",verbose=False)
sm.fit(f,P,freq_range=(1,40))
off,exp=sm.get_params("aperiodic")
ff=sm.data.freqs
data_log=sm.data.power_spectrum
full_log=sm.results.model.modeled_spectrum
ap_log=off-exp*np.log10(ff)
print(f"exponent={exp:.2f} offset={off:.2f}")

fig,ax=plt.subplots(figsize=(9.5,6.3))
ax.plot(ff,data_log,color=DARK,lw=2.6,label="observed spectrum",zorder=4)
ax.plot(ff,full_log,color=PINK,lw=2.3,label="full model fit",zorder=5)
ax.plot(ff,ap_log,color=PINK,lw=2.4,ls="--",label="aperiodic fit  (1/f)",zorder=3)
ax.fill_between(ff,ap_log,np.maximum(data_log,ap_log),color=LIGHT,alpha=.5,zorder=2)
ax.set_xscale("log"); ax.set_xlim(1,40)
ax.set_xticks([1,2,5,10,20,40]); ax.set_xticklabels([1,2,5,10,20,40])
ax.set_ylim(top=data_log.max()+0.7)
ax.set_xlabel("frequency (Hz, log scale)"); ax.set_ylabel("log₁₀ power")
ax.set_title("Deriving the aperiodic parameters (offset & exponent) from the PSD", pad=14, fontsize=15)

# OFFSET — intercept of the aperiodic fit
ax.scatter([1],[off],s=95,color=PINK,edgecolor=DARK,zorder=6)
ax.annotate("offset\n(vertical position\nof the 1/f fit)",xy=(1,off),xytext=(1.55,off+0.30),
            color=DARK,fontsize=12.5,fontweight="bold",
            arrowprops=dict(arrowstyle="-|>",color=DARK,lw=1.8))
# EXPONENT — slope of the aperiodic fit (slope triangle)
xa,xb=5.0,14.0
ya=off-exp*np.log10(xa); yb=off-exp*np.log10(xb)
ax.plot([xa,xb],[ya,ya],color=MUTE,lw=1.8)
ax.plot([xb,xb],[ya,yb],color=MUTE,lw=1.8)
ax.annotate(f"exponent χ = {exp:.2f}\n(steepness of the\n1/f slope)",
            xy=(xb,(ya+yb)/2),xytext=(15.5,ya+0.10),color=DARK,fontsize=12.5,fontweight="bold",
            arrowprops=dict(arrowstyle="-|>",color=DARK,lw=1.8))
ax.text(0.985,0.03,"specparam · fixed mode · fit 1–40 Hz",transform=ax.transAxes,
        ha="right",va="bottom",fontsize=10,color=MUTE)
ax.legend(loc="lower left",fontsize=11,framealpha=.92)
fig.savefig(f"{OUT}/22_specparam_decomposition.png",dpi=160,bbox_inches="tight",facecolor="white")
print("saved 22_specparam_decomposition.png")
