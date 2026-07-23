"""
Periodic (band-power) analyses mirroring the aperiodic investigation:
 A. HC-group comparison           (current / past / never)
 B. Cycle-phase mechanism         (naturally-cycling; and HC users)
 C. Formulation                   (combined vs progestin-only), age-adjusted
Measures: canonical (abs, rel) + specparam-adjusted (1/f removed).
Headline band = alpha (posterior + whole-head).
"""
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

DERIV="/sessions/vigilant-cool-goodall/mnt/ds007615/ds007615/derivatives/preproc"
BIDS="/sessions/vigilant-cool-goodall/mnt/ds007615/ds007615"; PH="/sessions/vigilant-cool-goodall/mnt/phenotype"
bp=pd.read_csv(f"{DERIV}/stats/bandpower_per_subject.csv",dtype={"subject":str})
# wide: one row per subject, columns per measure x acq
bp=bp.pivot(index="subject",columns="acq")
bp.columns=[f"{m}_{a}" for m,a in bp.columns]
def Ld(p,c):
    d=pd.read_csv(p,sep="\t",na_values="n/a"); d["subject"]=d["participant_id"].str.replace("sub-","",regex=False); return d.set_index("subject")[c]
meta=Ld(f"{BIDS}/participants.tsv",["age","group"]).join(Ld(f"{PH}/hc_usage.tsv",["main_prev","mens_phase"])).join(Ld(f"{PH}/lifestyle.tsv",["daily_nicotine","medication_use","alcohol_units"]))
d=bp.join(meta)
d["HCg"]=d["group"].map({1:"Current",2:"Past",3:"Never"})
d["HC"]=(d["group"]==1).astype(int)
d["combined"]=(d["main_prev"]==2).astype(float)
d["phase"]=d["mens_phase"].map({1:"Follicular",2:"Ovulatory",3:"Luteal"})
d["age"]=d["age"].astype(float)

# log-transform power measures (power is right-skewed)
def L(col):
    v=d[col].astype(float); return np.log10(v.replace(0,np.nan))
KEY=[("alpha_post_adj_ec","posterior adjusted alpha (EC)"),
     ("alpha_post_abs_ec","posterior canonical alpha (EC)"),
     ("adj_alpha_ec","whole-head adjusted alpha (EC)"),
     ("rel_alpha_ec","relative alpha (EC)"),
     ("adj_alpha_eo","whole-head adjusted alpha (EO)")]

def d_cohen(a,b):
    a,b=a[~np.isnan(a)],b[~np.isnan(b)]
    return (a.mean()-b.mean())/np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))

print("#"*64); print("A. HC-GROUP COMPARISON  (current / past / never)"); print("#"*64)
for col,lab in KEY:
    y=L(col) if col.startswith(("abs","adj","alpha_post_a")) else d[col].astype(float)
    g=[y[d.HCg==k].dropna().values for k in ["Current","Past","Never"]]
    F,p=stats.f_oneway(*g)
    cur=y[d.HC==1].dropna().values; non=y[d.HC==0].dropna().values
    t,pt=stats.ttest_ind(cur,non,nan_policy="omit")
    print(f"{lab:<34} ANOVA p={p:.3f} | current vs non p={pt:.3f} d={d_cohen(cur,non):+.2f}")

print("\n"+"#"*64); print("B. CYCLE-PHASE MECHANISM"); print("#"*64)
for grpname,mask in [("naturally-cycling",d.group.isin([2,3])),("HC users",d.group==1)]:
    sub=d[mask & d.phase.notna()]
    print(f"\n[{grpname}] n={len(sub)}  {dict(sub.phase.value_counts())}")
    for col,lab in [("alpha_post_adj_ec","posterior adj alpha EC"),("adj_alpha_ec","whole-head adj alpha EC"),("rel_alpha_ec","relative alpha EC")]:
        yv=np.log10(sub[col].replace(0,np.nan)) if col!="rel_alpha_ec" else sub[col]
        g=[yv[sub.phase==k].dropna().values for k in ["Follicular","Ovulatory","Luteal"]]
        if min(len(x) for x in g)<2: print(f"   {lab}: insufficient"); continue
        F,p=stats.f_oneway(*g); print(f"   {lab:<26} ANOVA p={p:.3f}")

print("\n"+"#"*64); print("C. FORMULATION (combined vs progestin-only), age-adjusted"); print("#"*64)
hcu=d[d.HC==1].copy(); print(f"n HC={len(hcu)} | combined={int(hcu.combined.sum())} progestin={int((hcu.combined==0).sum())}")
for band in ["delta","theta","alpha","beta"]:
    for meas,pref in [("adjusted","adj"),("relative","rel")]:
        col=f"{pref}_{band}_ec"
        yv=np.log10(hcu[col].replace(0,np.nan)) if pref=="adj" else hcu[col]
        hh=hcu.assign(y=yv).dropna(subset=["y","combined","age"])
        m=smf.ols("y ~ combined + age",data=hh).fit()
        b,p=m.params["combined"],m.pvalues["combined"]
        dd=d_cohen(hh.y[hh.combined==1].values,hh.y[hh.combined==0].values)
        star="*" if p<.05 else ""
        print(f"  {band:<6} {meas:<9} EC: b={b:+.3f} p={p:.3f} d={dd:+.2f} {star}")
print("\ndone")
