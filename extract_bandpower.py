"""
Extract canonical + specparam-adjusted (1/f-removed) band powers per subject.
Processes a chunk of subjects (argv: start end) and appends to bandpower_per_subject.csv.
"""
import sys, os, numpy as np, pandas as pd, mne
mne.set_log_level("ERROR")

DERIV="/sessions/vigilant-cool-goodall/mnt/ds007615/ds007615/derivatives/preproc"
OUTCSV=f"{DERIV}/stats/bandpower_per_subject.csv"
BANDS={"delta":(1,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30)}
TOTAL=(1,40)
POST=["O1","Oz","O2","PO7","PO3","POz","PO4","PO8","PO9","PO10","P1","P2","Pz","P3","P4","P5","P6","P7","P8"]

ap=pd.read_csv(f"{DERIV}/specparam/aperiodic_per_subject_channel.csv",dtype={"subject":str})

def bandpow(f,P):
    """P: (n_ch,n_freq) linear V^2/Hz. returns dict of per-channel band integrals (canonical)."""
    out={}
    for b,(lo,hi) in BANDS.items():
        m=(f>=lo)&(f<=hi); out[b]=np.trapz(P[:,m],f[m],axis=1)
    m=(f>=TOTAL[0])&(f<=TOTAL[1]); out["total"]=np.trapz(P[:,m],f[m],axis=1)
    return out

def run(subs):
    rows=[]
    for s in subs:
        for acq in ["ec","eo"]:
            fp=f"{DERIV}/sub-{s}/sub-{s}_task-rest_acq-{acq}_clean-epo.fif"
            if not os.path.exists(fp): continue
            ep=mne.read_epochs(fp)
            psd=ep.compute_psd(method="welch",fmin=1,fmax=45)
            P=psd.get_data().mean(0); f=np.asarray(psd.freqs); chs=list(psd.ch_names)
            # aperiodic per channel -> reconstruct 1/f and periodic residual
            a=ap[(ap.subject==s)&(ap.acq==acq)].set_index("channel")
            Pperi=np.zeros_like(P)
            for i,ch in enumerate(chs):
                if ch in a.index:
                    ap_lin=10**(a.loc[ch,"offset"]-a.loc[ch,"exponent"]*np.log10(f))
                    Pperi[i]=np.clip(P[i]-ap_lin,0,None)
                else:
                    Pperi[i]=np.nan
            can=bandpow(f,P); adj=bandpow(f,Pperi)
            post_idx=[i for i,c in enumerate(chs) if c in POST]
            row={"subject":s,"acq":acq,"n_ch":len(chs)}
            for b in BANDS:
                row[f"abs_{b}"]=np.nanmean(can[b])                       # canonical absolute (mean ch)
                row[f"rel_{b}"]=np.nanmean(can[b]/can["total"])          # canonical relative
                row[f"adj_{b}"]=np.nanmean(adj[b])                        # adjusted (1/f removed) absolute
            # posterior alpha (canonical + adjusted)
            row["alpha_post_abs"]=np.nanmean(can["alpha"][post_idx]) if post_idx else np.nan
            row["alpha_post_adj"]=np.nanmean(adj["alpha"][post_idx]) if post_idx else np.nan
            rows.append(row)
    df=pd.DataFrame(rows)
    hdr=not os.path.exists(OUTCSV)
    df.to_csv(OUTCSV,mode="a",header=hdr,index=False)
    print(f"wrote {len(df)} rows for subjects {subs[0]}..{subs[-1]}")

if __name__=="__main__":
    os.makedirs(f"{DERIV}/stats",exist_ok=True)
    allsubs=[f"{i:02d}" for i in range(1,70)]
    i0,i1=int(sys.argv[1]),int(sys.argv[2])
    run(allsubs[i0:i1])
