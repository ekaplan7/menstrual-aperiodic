"""
Mechanism test: does menstrual cycle phase move the aperiodic exponent
among naturally-cycling (non-current-HC) women?
"""
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

BIDS="/sessions/vigilant-cool-goodall/mnt/ds007615/ds007615"
PH="/sessions/vigilant-cool-goodall/mnt/phenotype"
ap=pd.read_csv(f"{BIDS}/derivatives/preproc/specparam/aperiodic_per_subject_channel.csv",dtype={"subject":str})
ap=ap[ap["acq"].isin(["ec","eo"])]
parts=pd.read_csv(f"{BIDS}/participants.tsv",sep="\t",na_values="n/a"); parts["subject"]=parts["participant_id"].str.replace("sub-","",regex=False); parts=parts.set_index("subject")
hc=pd.read_csv(f"{PH}/hc_usage.tsv",sep="\t",na_values="n/a"); hc["subject"]=hc["participant_id"].str.replace("sub-","",regex=False); hc=hc.set_index("subject")

# subject-level mean exponent/offset per condition
agg=ap.groupby(["subject","acq"])[["exponent","offset"]].mean().reset_index()
wide=agg.pivot(index="subject",columns="acq",values=["exponent","offset"])
wide.columns=[f"{p}_{a}" for p,a in wide.columns]
d=wide.join(parts[["age","group"]]).join(hc[["mens_phase","mens_phase_bin","cycle_phase","cycle_length"]])
# naturally cycling = non-current HC, observed phase
d=d[d["group"].isin([2,3]) & d["mens_phase"].notna()].copy()
d["phase"]=d["mens_phase"].map({1:"Follicular",2:"Ovulatory",3:"Luteal"})
d["luteal"]=(d["mens_phase"]==3).astype(int)
print(f"N naturally-cycling with phase = {len(d)}")
print(d["phase"].value_counts(),"\n")

def eta2(groups):
    allv=np.concatenate(groups); gm=allv.mean()
    ssb=sum(len(g)*(g.mean()-gm)**2 for g in groups); sst=((allv-gm)**2).sum()
    return ssb/sst

for dv in ["exponent_ec","exponent_eo","offset_ec","offset_eo"]:
    print("="*60); print(dv.upper())
    g=[d.loc[d.phase==p,dv].values for p in ["Follicular","Ovulatory","Luteal"]]
    for p,gr in zip(["Follicular","Ovulatory","Luteal"],g):
        print(f"  {p:<11} n={len(gr):2d}  mean={gr.mean():.3f}  sd={gr.std(ddof=1):.3f}")
    F,pF=stats.f_oneway(*g)
    print(f"  1-way ANOVA (3 phases):  F={F:.2f}  p={pF:.3f}  eta^2={eta2(g):.3f}")
    # luteal vs first-half (follicular+ovulatory)
    lut=d.loc[d.luteal==1,dv].values; fh=d.loc[d.luteal==0,dv].values
    t,pt=stats.ttest_ind(lut,fh)
    ds=(lut.mean()-fh.mean())/np.sqrt(((len(lut)-1)*lut.var(ddof=1)+(len(fh)-1)*fh.var(ddof=1))/(len(lut)+len(fh)-2))
    print(f"  luteal(n={len(lut)}) vs first-half(n={len(fh)}): t={t:.2f} p={pt:.3f}  Cohen d={ds:+.2f}")
    # ANCOVA controlling age
    m=smf.ols(f"{dv} ~ C(phase) + age",data=d).fit()
    print(f"  ANCOVA + age: phase joint p={m.f_test('C(phase)[T.Luteal]=0, C(phase)[T.Ovulatory]=0').pvalue:.3f}  (age p={m.pvalues['age']:.3f})")
    # cyclic day-of-cycle regression
    dc=d.dropna(subset=["cycle_phase","cycle_length"]).copy()
    dc=dc[dc.cycle_length>0]
    th=2*np.pi*dc.cycle_phase/dc.cycle_length
    dc["sin"],dc["cos"]=np.sin(th),np.cos(th)
    mc=smf.ols(f"{dv} ~ sin + cos + age",data=dc).fit()
    pj=mc.f_test("sin=0, cos=0").pvalue
    print(f"  cyclic day-of-cycle (n={len(dc)}): sin/cos joint p={float(pj):.3f}  R2_model={mc.rsquared:.2f}")
print("="*60)
print("done")
