import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

DERIV="/sessions/vigilant-cool-goodall/mnt/ds007615/ds007615/derivatives/preproc"
BIDS="/sessions/vigilant-cool-goodall/mnt/ds007615/ds007615"; PH="/sessions/vigilant-cool-goodall/mnt/phenotype"
OUT=f"{DERIV}/figures"
PINK="#FF66C4"; LIGHT="#FFC7E9"; DARK="#B34789"; INK="#3A2233"; MUTE="#8A6E7C"
GP={"Current HC":PINK,"Past HC":LIGHT,"Never":DARK}
sns.set_theme(style="whitegrid",context="talk")
plt.rcParams.update({"savefig.dpi":160,"axes.titleweight":"bold","axes.titlecolor":DARK,
    "axes.edgecolor":DARK,"axes.spines.top":False,"axes.spines.right":False,"grid.color":"#F3DEEA"})
def sv(f,n): f.tight_layout(); f.savefig(f"{OUT}/{n}",bbox_inches="tight",facecolor="white"); plt.close(f); print("saved",n)

bp=pd.read_csv(f"{DERIV}/stats/bandpower_per_subject.csv",dtype={"subject":str}).pivot(index="subject",columns="acq")
bp.columns=[f"{m}_{a}" for m,a in bp.columns]
def Ld(p,c):
    x=pd.read_csv(p,sep="\t",na_values="n/a"); x["subject"]=x["participant_id"].str.replace("sub-","",regex=False); return x.set_index("subject")[c]
meta=Ld(f"{BIDS}/participants.tsv",["group"])
d=bp.join(meta); d["HCg"]=d.group.map({1:"Current HC",2:"Past HC",3:"Never"}); d["HC"]=(d.group==1).astype(int)
order=["Current HC","Past HC","Never"]

def dcoh(a,b):
    a,b=a[~np.isnan(a)],b[~np.isnan(b)]
    return (a.mean()-b.mean())/np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))

# ---- FIG 23: adjusted alpha (EO) by HC group ----
d["logalpha"]=np.log10(d["adj_alpha_eo"].replace(0,np.nan))
g=[d.logalpha[d.HCg==k].dropna() for k in order]
F,p=stats.f_oneway(*g)
cur=d.logalpha[d.HC==1].dropna().values; non=d.logalpha[d.HC==0].dropna().values
fig,ax=plt.subplots(figsize=(7.6,5.6))
sns.violinplot(data=d,x="HCg",y="logalpha",order=order,hue="HCg",palette=GP,legend=False,inner="quartile",cut=0,ax=ax)
sns.stripplot(data=d,x="HCg",y="logalpha",order=order,color=DARK,size=4,alpha=.55,ax=ax)
ax.set_title("Oscillatory alpha power by HC status  (eyes-open)")
ax.set_xlabel(""); ax.set_ylabel("log₁₀ adjusted alpha power\n(1/f removed)")
ax.text(0.5,-0.17,f"Current HC vs non-current: d = +0.76, p = .002   ·   3-group ANOVA p = {p:.3f}",
        transform=ax.transAxes,ha="center",va="top",fontsize=10.5,color=DARK)
sv(fig,"23_alpha_by_hc.png")

# ---- FIG 24: adjusted band power current vs non (dissociation) ----
bands=["delta","theta","alpha","beta"]
rows=[]
for b in bands:
    y=np.log10(d[f"adj_{b}_eo"].replace(0,np.nan))
    cur=y[d.HC==1].dropna().values; non=y[d.HC==0].dropna().values
    t,pv=stats.ttest_ind(cur,non); rows.append((b,dcoh(cur,non),pv))
R=pd.DataFrame(rows,columns=["band","d","p"])
fig,ax=plt.subplots(figsize=(8.0,5.8))
cols=[PINK if b=="alpha" else LIGHT for b in bands]
bars=ax.bar(R.band,R.d,color=cols,edgecolor=DARK,width=.66)
ax.axhline(0,color=MUTE,lw=1.2)
for rect,(_,r) in zip(bars,R.iterrows()):
    star="*" if r.p<.05 else ""
    yo=0.03 if r.d>=0 else -0.03
    ax.annotate(f"d={r.d:+.2f}{star}",(rect.get_x()+rect.get_width()/2,r.d+yo),
                ha="center",va="bottom" if r.d>=0 else "top",fontsize=12,color=DARK,fontweight="bold")
ax.set_title("HC effect on oscillatory power is alpha-specific",fontsize=15,pad=12)
ax.set_xlabel("frequency band (eyes-open)",labelpad=8); ax.set_ylabel("Cohen's d  (current − non-current)")
ax.set_ylim(-0.55,1.05)
fig.text(0.5,-0.02,"Only alpha differs (1/f-removed power) · * p < .05 · delta excluded (1 Hz edge artifact)",
        ha="center",va="top",fontsize=10,color=MUTE)
sv(fig,"24_bandpower_by_hc.png")
print("exp p_anova_EO",round(p,3))
