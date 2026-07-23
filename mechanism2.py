"""
Expanded hormonal / menstrual analysis of the aperiodic parameters.
  A. Cycle phase -> exponent in HC users (hormonally-clamped negative control) + phase x HC interaction
  B. Within HC users: formulation (combined vs progestin-only), delivery, duration of use
  C. Omnibus: what predicts the exponent across all women (hormonal status + known drivers)
"""
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

BIDS="/sessions/vigilant-cool-goodall/mnt/ds007615/ds007615"; PH="/sessions/vigilant-cool-goodall/mnt/phenotype"
ap=pd.read_csv(f"{BIDS}/derivatives/preproc/specparam/aperiodic_per_subject_channel.csv",dtype={"subject":str})
ap=ap[ap["acq"].isin(["ec","eo"])]
def L(p,cols):
    d=pd.read_csv(p,sep="\t",na_values="n/a"); d["subject"]=d["participant_id"].str.replace("sub-","",regex=False)
    return d.set_index("subject")[cols]
parts=L(f"{BIDS}/participants.tsv",["age","group"])
hc=L(f"{PH}/hc_usage.tsv",["main_prev","oc_nonoc","prev_duration","mens_phase","cycle_phase","cycle_length"])
life=L(f"{PH}/lifestyle.tsv",["medication_use","daily_nicotine","alcohol_units"])

agg=ap.groupby(["subject","acq"])[["exponent","offset"]].mean().reset_index()
w=agg.pivot(index="subject",columns="acq",values=["exponent","offset"]); w.columns=[f"{p}_{a}" for p,a in w.columns]
d=w.join(parts).join(hc).join(life)
d["HC"]=(d["group"]==1).astype(int)
d["phase"]=d["mens_phase"].map({1:"Follicular",2:"Ovulatory",3:"Luteal"})
d["med"]=(d["medication_use"]==1).astype(int)
d["nic"]=(d["daily_nicotine"]==1).astype(int)
d["alc"]=d["alcohol_units"].astype(float); d["alc"]=d["alc"].fillna(d["alc"].median())
d["combined"]=(d["main_prev"]==2).astype("float")     # 2=combined,1=progestin-only
d["nonoral"]=(d["oc_nonoc"]==2).astype("float")
def z(s): return (s-s.mean())/s.std()

DVS=["exponent_ec","exponent_eo","offset_ec","offset_eo"]

print("#"*66)
print("A. CYCLE PHASE -> EXPONENT  (HC users = clamped negative control)")
print("#"*66)
hcp=d[(d.HC==1)&d.phase.notna()]
print(f"HC users with reported phase: n={len(hcp)}  ",dict(hcp.phase.value_counts()))
for dv in ["exponent_ec","exponent_eo"]:
    g=[hcp.loc[hcp.phase==p,dv].values for p in ["Follicular","Ovulatory","Luteal"]]
    F,p=stats.f_oneway(*g)
    print(f"  {dv}: ANOVA F={F:.2f} p={p:.3f}  (means "
          + ", ".join(f"{ph}={gr.mean():.3f}" for ph,gr in zip(['Fol','Ovu','Lut'],g))+")")
# phase x HC interaction (cyclers + HC users with phase)
both=d[d.phase.notna()].copy()
print(f"\nphase x HC interaction (all with phase, n={len(both)}):")
for dv in ["exponent_ec","exponent_eo"]:
    m=smf.ols(f"{dv} ~ C(phase)*HC + age",data=both).fit()
    ix=[t for t in m.pvalues.index if ":" in t]
    pj=m.f_test(", ".join(f"{t}=0" for t in ix)).pvalue
    print(f"  {dv}: phase x HC interaction joint p={float(pj):.3f}")

print("\n"+"#"*66)
print("B. WITHIN HC USERS: formulation, delivery, duration (control age)")
print("#"*66)
hcu=d[d.HC==1].copy()
print(f"n HC = {len(hcu)} | combined={int(hcu.combined.sum())} progestin-only={int((hcu.combined==0).sum())}"
      f" | non-oral={int(hcu.nonoral.sum())} oral={int((hcu.nonoral==0).sum())}")
for dv in DVS:
    m=smf.ols(f"{dv} ~ combined + age",data=hcu).fit()
    b,p=m.params["combined"],m.pvalues["combined"]
    lut=hcu.loc[hcu.combined==1,dv]; fh=hcu.loc[hcu.combined==0,dv]
    dd=(lut.mean()-fh.mean())/np.sqrt(((len(lut)-1)*lut.var()+ (len(fh)-1)*fh.var())/(len(lut)+len(fh)-2))
    md=smf.ols(f"{dv} ~ prev_duration + age",data=hcu).fit()
    mo=smf.ols(f"{dv} ~ nonoral + age",data=hcu).fit()
    print(f"  {dv}: combined vs progestin b={b:+.3f} p={p:.3f} d={dd:+.2f} | "
          f"duration p={md.pvalues['prev_duration']:.3f} | oral/nonoral p={mo.pvalues['nonoral']:.3f}")

print("\n"+"#"*66)
print("C. OMNIBUS: predictors of the exponent across ALL women (n=%d)"%len(d))
print("   exponent ~ HC + age + medication + nicotine + alcohol   (standardized)")
print("#"*66)
dd=d.copy()
dd["age_z"]=z(dd.age); dd["alc_z"]=z(dd.alc)
for dv in DVS:
    dd["y"]=z(dd[dv])
    m=smf.ols("y ~ HC + age_z + med + nic + alc_z",data=dd).fit()
    print(f"\n {dv}  (model R2={m.rsquared:.2f})")
    for term in ["HC","age_z","med","nic","alc_z"]:
        star="***" if m.pvalues[term]<.001 else "**" if m.pvalues[term]<.01 else "*" if m.pvalues[term]<.05 else ""
        print(f"    {term:<7} beta={m.params[term]:+.3f}  p={m.pvalues[term]:.3f} {star}")
print("\ndone")
