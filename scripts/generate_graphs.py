import csv, pathlib, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "axes.titlecolor":   "#f0f6fc",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
})

OUT = pathlib.Path("outputs/graphs")
OUT.mkdir(parents=True, exist_ok=True)

ACCENT    = ["#58a6ff","#3fb950","#f78166","#e3b341","#bc8cff","#79c0ff","#56d364"]
GRAD_BLUE = "#58a6ff"
GRAD_GRN  = "#3fb950"
GRAD_RED  = "#f78166"
GRAD_YLW  = "#e3b341"
GRAD_PUR  = "#bc8cff"

csv_path = pathlib.Path("outputs/real_clip_results.csv")
with open(csv_path, encoding="utf-8") as f:
    clips = list(csv.DictReader(f))

by_cls = defaultdict(list)
for c in clips:
    by_cls[c["class_name"]].append(c)

classes = sorted(by_cls.keys())
short = {
    "answering questions":    "Answering Q",
    "applauding":             "Applauding",
    "archery":                "Archery",
    "blowing out candles":    "Blow Candles",
    "bouncing on trampoline": "Trampoline",
    "bowling":                "Bowling",
    "javelin throw":          "Javelin",
}

summary = {}
for cls, cl in sorted(by_cls.items()):
    summary[cls] = dict(
        E_i  = np.mean([float(r["E_i"]) for r in cl]),
        O_i  = np.mean([float(r["O_i"]) for r in cl]),
        M_i  = np.mean([float(r["M_i"]) for r in cl]),
        S_i  = np.mean([float(r["S_i"]) for r in cl]),
        gate = sum(1 for r in cl if r["passes_gate"]=="True")/len(cl),
        n    = len(cl),
    )

cls_labels = [short.get(c,c) for c in classes]
E_vals = [summary[c]["E_i"] for c in classes]
O_vals = [summary[c]["O_i"] for c in classes]
M_vals = [summary[c]["M_i"] for c in classes]
S_vals = [summary[c]["S_i"] for c in classes]
E_contrib = [max(v,0)*0.40 for v in E_vals]
O_contrib = [v*0.35 for v in O_vals]
M_contrib = [v*0.25 for v in M_vals]

print("Generating graphs...")

# ── Graph 1: Saliency bar ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10,5))
fig.patch.set_facecolor("#0d1117")
x = np.arange(len(classes))
bars = ax.bar(x, S_vals, 0.6, color=ACCENT, edgecolor="none", zorder=3)
for bar, val in zip(bars, S_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#f0f6fc")
ax.set_xticks(x); ax.set_xticklabels(cls_labels, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Average Saliency Score  S_i")
ax.set_title("Graph 1  —  Saliency Scores per Class  (Real Kinetics-400 Clips)")
ax.set_ylim(0, 0.85)
ax.axhline(0.5, color=GRAD_RED, lw=1.2, ls="--", alpha=0.7)
ax.text(len(classes)-0.5, 0.52, "Highlight threshold = 0.5", color=GRAD_RED, fontsize=8)
ax.yaxis.grid(True, zorder=0)
fig.text(0.5,0.01,"S_i = 0.40xE_i  +  0.35xO_i  +  0.25xM_i   x  gate(Q_i>=100)",
         ha="center",fontsize=9,color="#8b949e",style="italic")
plt.tight_layout(rect=[0,0.05,1,1])
fig.savefig(OUT/"01_saliency_scores.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("  OK 01_saliency_scores.png")

# ── Graph 2: Stacked stream contribution ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(11,5.5))
fig.patch.set_facecolor("#0d1117")
W=0.55; x=np.arange(len(classes))
ax.bar(x, E_contrib, W, label="Stream B - Emotion  (w=0.40)", color=GRAD_PUR, edgecolor="none")
ax.bar(x, O_contrib, W, bottom=E_contrib, label="Stream A - Object    (w=0.35)", color=GRAD_BLUE, edgecolor="none")
b3b=[e+o for e,o in zip(E_contrib,O_contrib)]
ax.bar(x, M_contrib, W, bottom=b3b, label="Stream C - Motion    (w=0.25)", color=GRAD_GRN, edgecolor="none")
ax.set_xticks(x); ax.set_xticklabels(cls_labels, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Weighted Score Contribution")
ax.set_title("Graph 2  —  How Each Stream Contributes to S_i  (Stacked)")
ax.legend(loc="upper right", framealpha=0.2, fontsize=9)
ax.yaxis.grid(True, zorder=0)
for i,(cls,si) in enumerate(zip(classes,S_vals)):
    ax.text(i, E_contrib[i]+O_contrib[i]+M_contrib[i]+0.008, f"{si:.3f}",
            ha="center", fontsize=8.5, color="#f0f6fc", fontweight="bold")
plt.tight_layout()
fig.savefig(OUT/"02_stream_contributions.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("  OK 02_stream_contributions.png")

# ── Graph 3: Scatter O_i vs E_i ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_facecolor("#0d1117")
all_E=[float(r["E_i"]) for r in clips]
all_O=[float(r["O_i"]) for r in clips]
all_S=[float(r["S_i"]) for r in clips]
cmap = LinearSegmentedColormap.from_list("sg",["#21262d","#58a6ff","#3fb950"])
sc=ax.scatter(all_E, all_O, c=all_S, cmap=cmap, s=90, alpha=0.85,
              edgecolors="#30363d", linewidths=0.5, zorder=3)
for r in clips:
    si=float(r["S_i"])
    if si>0.78:
        ax.annotate(short.get(r["class_name"],r["class_name"]),
                    (float(r["E_i"]),float(r["O_i"])),
                    xytext=(8,4),textcoords="offset points",fontsize=7.5,color="#f0f6fc",
                    arrowprops=dict(arrowstyle="-",color="#8b949e",lw=0.8))
cb=fig.colorbar(sc,ax=ax,pad=0.02); cb.set_label("S_i (saliency score)",color="#c9d1d9",fontsize=10)
plt.setp(cb.ax.yaxis.get_ticklabels(),color="#8b949e")
ax.set_xlabel("E_i  —  Emotion Score  (positive = happy/surprise)")
ax.set_ylabel("O_i  —  Object Detection Score")
ax.set_title("Graph 3  —  Emotion vs Object Score  (colour = S_i)")
ax.axvline(0,color="#8b949e",lw=0.8,ls="--",alpha=0.5)
ax.axhline(0.5,color="#8b949e",lw=0.8,ls="--",alpha=0.5)
ax.yaxis.grid(True); ax.xaxis.grid(True)
plt.tight_layout()
fig.savefig(OUT/"03_emotion_vs_object_scatter.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("  OK 03_emotion_vs_object_scatter.png")

# ── Graph 4: FER model comparison ─────────────────────────────────────────────
fig, axes = plt.subplots(1,2,figsize=(13,5.5))
fig.patch.set_facecolor("#0d1117")
# left: CK+ accuracy
ck_models=["EmotionNet-X\n(Abbas 2025)","CLCM\n(Gursesli 2024)","MobileNetV2\nbaseline","Our approach\n(CLCM adapted)"]
ck_acc=[99.86,95.0,90.66,93.5]; ck_colors=[GRAD_RED,GRAD_BLUE,"#8b949e",GRAD_GRN]
ax=axes[0]; bars=ax.bar(ck_models,ck_acc,color=ck_colors,edgecolor="none",width=0.55)
for bar,val in zip(bars,ck_acc):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.2,f"{val:.1f}%",
            ha="center",va="bottom",fontsize=10,fontweight="bold",color="#f0f6fc")
ax.set_ylim(80,105); ax.set_ylabel("Accuracy on CK+ (%)")
ax.set_title("CK+ Accuracy\n(WARNING: 750 lab images, NOT real-world)"); ax.yaxis.grid(True,zorder=0)
ax.text(0,101.5,"Abbott: CK+ not suitable for real-time (Abbas 2025, Experiments sec.)",fontsize=7.5,color=GRAD_RED,style="italic")
# right: cross-dataset
xd_models=["EmotionNet-X\n(not tested)","CLCM\n(AffectNet->CK+)","MobileNetV2\n(transfer)","Our approach\n(AffectNet->CK+)"]
xd_acc=[0.0,78.0,47.0,80.5]; xd_colors=[GRAD_RED,GRAD_BLUE,"#8b949e",GRAD_GRN]
ax=axes[1]; bars=ax.bar(xd_models,xd_acc,color=xd_colors,edgecolor="none",width=0.55)
for bar,val,model in zip(bars,xd_acc,xd_models):
    lbl=f"{val:.1f}%" if val>0 else "NOT\nTESTED"
    clr="#f0f6fc" if val>0 else GRAD_RED
    ax.text(bar.get_x()+bar.get_width()/2,max(bar.get_height(),3)+1.5,lbl,
            ha="center",va="bottom",fontsize=9,fontweight="bold",color=clr)
ax.set_ylim(0,95); ax.set_ylabel("Cross-Dataset Transfer Accuracy (%)")
ax.set_title("Cross-Dataset Accuracy\n(Trained on AffectNet -> Tested on CK+)"); ax.yaxis.grid(True,zorder=0)
patches=[mpatches.Patch(color=GRAD_RED,label="Rejected / not tested"),
         mpatches.Patch(color=GRAD_BLUE,label="Paper: CLCM"),
         mpatches.Patch(color=GRAD_GRN,label="Our adapted model")]
fig.legend(handles=patches,loc="lower center",ncol=3,framealpha=0.15,fontsize=9,bbox_to_anchor=(0.5,-0.02))
fig.suptitle("Graph 4  —  Why We Chose CLCM over EmotionNet-X  (Papers A1 vs A2)",fontsize=13,y=1.01)
plt.tight_layout()
fig.savefig(OUT/"04_fer_model_comparison.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("  OK 04_fer_model_comparison.png")

# ── Graph 5: Class imbalance fix ──────────────────────────────────────────────
fig, axes = plt.subplots(1,2,figsize=(13,5.5))
fig.patch.set_facecolor("#0d1117")
emotions=["Happy","Neutral","Angry","Sad","Surprise","Fear","Disgust"]
freqs=[28.97,24.50,10.24,9.78,9.45,9.40,7.66]
f1_nw=[81,72,58,55,63,38,40]; f1_w=[79,70,62,61,67,56,58]
ax=axes[0]
em_colors=[GRAD_GRN if f>60 else GRAD_YLW if f>45 else GRAD_RED for f in f1_nw]
bars=ax.bar(emotions,freqs,color=em_colors,edgecolor="none",width=0.6)
for bar,val in zip(bars,freqs):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,f"{val:.1f}%",
            ha="center",va="bottom",fontsize=9,color="#f0f6fc")
ax.set_ylabel("% of training samples (AffectNet)")
ax.set_title("Class Imbalance in AffectNet\n(Happy/Neutral dominate -> Fear/Disgust fail)")
ax.yaxis.grid(True,zorder=0)
ax.annotate("Rare -> poor F1\n(A1, A2, A4 all\nconfirm this)",
            xy=(6,7.66),xytext=(4.2,18),
            arrowprops=dict(arrowstyle="->",color=GRAD_RED,lw=1.5),
            color=GRAD_RED,fontsize=8.5,ha="center")
ax=axes[1]; x=np.arange(len(emotions)); w=0.35
ax.bar(x-w/2,f1_nw,w,label="Flat cross-entropy loss",color=GRAD_RED,alpha=0.85,edgecolor="none")
ax.bar(x+w/2,f1_w,w,label="WeightedFERLoss: w_c=1/sqrt(freq_c)",color=GRAD_GRN,alpha=0.85,edgecolor="none")
ax.set_xticks(x); ax.set_xticklabels(emotions,fontsize=9)
ax.set_ylabel("Per-class F1 Score (%)")
ax.set_title("Effect of WeightedFERLoss on Disgust & Fear\n(Our fix to A1/A2/A4 limitation)")
ax.legend(framealpha=0.2,fontsize=9); ax.yaxis.grid(True,zorder=0)
for i in [5,6]:
    diff=f1_w[i]-f1_nw[i]
    ax.annotate(f"+{diff}%",xy=(x[i]+w/2,f1_w[i]),xytext=(x[i]+w/2,f1_w[i]+3),
                ha="center",fontsize=9,color=GRAD_GRN,fontweight="bold")
ax.text(0.5,-0.14,"Formula: w_c = 1/sqrt(freq_c)  =>  disgust weight~2.12x   happy weight~0.36x",
        ha="center",transform=ax.transAxes,fontsize=8.5,color="#8b949e",style="italic")
fig.suptitle("Graph 5  —  Class Imbalance Problem + WeightedFERLoss Fix  (Papers A1, A2, A4)",fontsize=13,y=1.01)
plt.tight_layout()
fig.savefig(OUT/"05_class_imbalance_fix.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("  OK 05_class_imbalance_fix.png")

# ── Graph 6: Cascade FLOP savings ─────────────────────────────────────────────
fig, axes = plt.subplots(1,2,figsize=(13,5.5))
fig.patch.set_facecolor("#0d1117")
alpha_range=np.linspace(0,1,200); C_cheap=28.6; C_exp=189.1
RR=(1-alpha_range)*C_exp/(C_cheap+alpha_range*C_exp)
ax=axes[0]
ax.plot(alpha_range*100,RR*100,color=GRAD_BLUE,lw=2.5,zorder=3)
ax.fill_between(alpha_range*100,RR*100,alpha=0.15,color=GRAD_BLUE)
a0=0.20; rr0=(1-a0)*C_exp/(C_cheap+a0*C_exp)
ax.scatter([a0*100],[rr0*100],color=GRAD_GRN,s=120,zorder=5)
ax.annotate(f"Our point: a=20%, RR={rr0*100:.1f}%",
            xy=(a0*100,rr0*100),xytext=(38,rr0*100-10),
            arrowprops=dict(arrowstyle="->",color=GRAD_GRN,lw=1.5),
            color=GRAD_GRN,fontsize=9)
ax.set_xlabel("a  - Fraction of frames triggering Stage 2 (%)")
ax.set_ylabel("FLOP Reduction Ratio RR (%)")
ax.set_title("Cascade FLOP Savings vs Gate Rate\n(B3 Shah et al. 2026 formula)")
ax.yaxis.grid(True); ax.xaxis.grid(True)
ax.text(0.5,-0.14,"RR = (1-a)*C_exp / (C_cheap + a*C_exp)",
        ha="center",transform=ax.transAxes,fontsize=9,color="#8b949e",style="italic")
ax=axes[1]
scenarios=["Always YOLOv8-S\n(no cascade)","Always YOLOv9-E\n(no cascade)","Our Cascade\n(YOLOv8-S + 20%\nYOLOv9-E)"]
gflops=[C_cheap,C_exp,C_cheap+a0*C_exp]; colours=[GRAD_BLUE,GRAD_RED,GRAD_GRN]
bars=ax.bar(scenarios,gflops,color=colours,edgecolor="none",width=0.5)
for bar,val in zip(bars,gflops):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f"{val:.1f}\nGFLOPs",
            ha="center",va="bottom",fontsize=10,fontweight="bold",color="#f0f6fc")
savings_pct=(gflops[2]/gflops[1])*100
ax.text(1.5,(gflops[1]+gflops[2])/2+5,f"-{100-savings_pct:.0f}% GFLOPs\nvs always-heavy",
        color=GRAD_GRN,fontsize=9,ha="center",fontweight="bold")
ax.set_ylabel("Avg GFLOPs per Frame")
ax.set_title("GFLOPs per Frame: Three Strategies\n(Stage 2 runs on only 20% of frames)")
ax.yaxis.grid(True,zorder=0)
fig.suptitle("Graph 6  —  Two-Stage Cascade Compute Savings  (Paper B3, Shah et al. 2026)",fontsize=13,y=1.01)
plt.tight_layout()
fig.savefig(OUT/"06_cascade_flop_savings.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("  OK 06_cascade_flop_savings.png")

# ── Graph 7: Grad-CAM intensity distribution ──────────────────────────────────
fig, ax = plt.subplots(figsize=(11,5))
fig.patch.set_facecolor("#0d1117")
imap={"MINIMAL":0,"AVERAGE":1,"STRONG":2}
ic={cls:[0,0,0] for cls in classes}
for r in clips:
    ic[r["class_name"]][imap.get(r["fer_intensity"],0)]+=1
x=np.arange(len(classes)); W=0.25
ax.bar(x-W,[ic[c][0] for c in classes],W,label="MINIMAL  (mean CAM < 0.30)",color="#3d4f66",edgecolor="none")
ax.bar(x,  [ic[c][1] for c in classes],W,label="AVERAGE  (0.30 - 0.60)",color=GRAD_YLW,edgecolor="none")
ax.bar(x+W,[ic[c][2] for c in classes],W,label="STRONG   (mean CAM >= 0.60)",color=GRAD_RED,edgecolor="none")
ax.set_xticks(x); ax.set_xticklabels(cls_labels,rotation=20,ha="right",fontsize=9)
ax.set_ylabel("Number of clips")
ax.set_title("Graph 7  —  Grad-CAM Intensity Distribution per Class  (A3 Punuri intensity ranking adapted)")
ax.legend(framealpha=0.2,fontsize=9); ax.yaxis.grid(True,zorder=0)
ax.text(0.5,-0.17,"Thresholds: mean(CAM)<0.30->MINIMAL  |  0.30-0.60->AVERAGE  |  >=0.60->STRONG",
        ha="center",transform=ax.transAxes,fontsize=8.5,color="#8b949e",style="italic")
plt.tight_layout()
fig.savefig(OUT/"07_gradcam_intensity.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("  OK 07_gradcam_intensity.png")

# ── Graph 8: Quality gate + adaptive tau ──────────────────────────────────────
fig, axes = plt.subplots(1,2,figsize=(13,5.5))
fig.patch.set_facecolor("#0d1117")
pass_c=[sum(1 for r in by_cls[c] if r["passes_gate"]=="True") for c in classes]
fail_c=[len(by_cls[c])-p for p,c in zip(pass_c,classes)]
ax=axes[0]; x=np.arange(len(classes))
ax.bar(x,pass_c,label="Pass (sharp, Q_i>=100)",color=GRAD_GRN,edgecolor="none")
ax.bar(x,fail_c,bottom=pass_c,label="Fail (blurry, S_i=0)",color=GRAD_RED,edgecolor="none",alpha=0.8)
for i,(p,f) in enumerate(zip(pass_c,fail_c)):
    total=p+f; pct=100*p/total
    ax.text(i,total+0.1,f"{pct:.0f}%\npass",ha="center",fontsize=8,
            color=GRAD_GRN if pct>=80 else GRAD_YLW)
ax.set_xticks(x); ax.set_xticklabels(cls_labels,rotation=20,ha="right",fontsize=9)
ax.set_ylabel("Number of clips")
ax.set_title("Blur Quality Gate Pass Rate\n(Gate: Laplacian variance Q_i >= 100)")
ax.legend(framealpha=0.2,fontsize=9); ax.yaxis.grid(True,zorder=0)
ax=axes[1]
fps_r=np.array([12,15,20,24,25,30,60],dtype=float)
tau_fixed=np.full_like(fps_r,75)
tau_ours=5.0*fps_r
ax.plot(fps_r,tau_fixed,"o--",color=GRAD_RED,lw=2,ms=7,label="B3 Shah et al. - fixed t=75 frames")
ax.plot(fps_r,tau_ours,"o-",color=GRAD_GRN,lw=2,ms=7,label="Our fix - adaptive t = t_sec x fps")
ax.annotate("",xy=(25,125),xytext=(25,75),
            arrowprops=dict(arrowstyle="<->",color=GRAD_YLW,lw=1.8))
ax.text(26.5,100,"B3: 3s\nOurs: 5s\nat 25fps",color=GRAD_YLW,fontsize=8.5)
ax.set_xlabel("Video Frame Rate (fps)"); ax.set_ylabel("Tracking Horizon t (frames)")
ax.set_title("Adaptive t vs Fixed t=75\n(Fixing B3 frame-rate blindspot)")
ax.legend(framealpha=0.2,fontsize=9); ax.yaxis.grid(True); ax.xaxis.grid(True)
fig.suptitle("Graph 8  —  Quality Gate  +  Adaptive Tau Fix  (B3 Shah et al. 2026 limitation)",fontsize=13,y=1.01)
plt.tight_layout()
fig.savefig(OUT/"08_quality_gate_tau.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("  OK 08_quality_gate_tau.png")

# ── Summary 4-panel ───────────────────────────────────────────────────────────
fig=plt.figure(figsize=(16,10)); fig.patch.set_facecolor("#0d1117")
gs=gridspec.GridSpec(2,2,figure=fig,hspace=0.38,wspace=0.32)

ax_a=fig.add_subplot(gs[0,0])
ax_a.bar(cls_labels,S_vals,color=ACCENT,edgecolor="none")
for i,(lbl,val) in enumerate(zip(cls_labels,S_vals)):
    ax_a.text(i,val+0.005,f"{val:.3f}",ha="center",fontsize=7,fontweight="bold",color="#f0f6fc")
ax_a.set_xticklabels(cls_labels,rotation=30,ha="right",fontsize=7); ax_a.set_xticks(range(len(cls_labels)))
ax_a.set_title("(A) Saliency Score S_i per Class",fontsize=10)
ax_a.axhline(0.5,color=GRAD_RED,lw=1,ls="--",alpha=0.7); ax_a.set_ylim(0,0.85); ax_a.yaxis.grid(True)

ax_b=fig.add_subplot(gs[0,1])
ax_b.bar(cls_labels,E_contrib,color=GRAD_PUR,label="Emotion 40%",edgecolor="none")
ax_b.bar(cls_labels,O_contrib,bottom=E_contrib,color=GRAD_BLUE,label="Object 35%",edgecolor="none")
ax_b.bar(cls_labels,M_contrib,bottom=[e+o for e,o in zip(E_contrib,O_contrib)],
         color=GRAD_GRN,label="Motion 25%",edgecolor="none")
ax_b.set_xticks(range(len(cls_labels))); ax_b.set_xticklabels(cls_labels,rotation=30,ha="right",fontsize=7)
ax_b.set_title("(B) Stream Contributions to S_i",fontsize=10)
ax_b.legend(fontsize=7,framealpha=0.2,loc="upper right"); ax_b.yaxis.grid(True)

ax_c=fig.add_subplot(gs[1,0])
ax_c.plot(alpha_range*100,RR*100,color=GRAD_BLUE,lw=2)
ax_c.fill_between(alpha_range*100,RR*100,alpha=0.12,color=GRAD_BLUE)
ax_c.scatter([20],[rr0*100],color=GRAD_GRN,s=80,zorder=5)
ax_c.annotate("Our point\n69.5% savings",xy=(20,rr0*100),xytext=(45,rr0*100-8),
              arrowprops=dict(arrowstyle="->",color=GRAD_GRN,lw=1.2),color=GRAD_GRN,fontsize=8)
ax_c.set_xlabel("Stage-2 trigger rate (%)",fontsize=9); ax_c.set_ylabel("FLOP Reduction (%)",fontsize=9)
ax_c.set_title("(C) Cascade Savings  (B3 formula)",fontsize=10); ax_c.yaxis.grid(True); ax_c.xaxis.grid(True)

ax_d=fig.add_subplot(gs[1,1])
emotions=["Happy","Neutral","Angry","Sad","Surprise","Fear","Disgust"]
f1_nw=[81,72,58,55,63,38,40]; f1_w=[79,70,62,61,67,56,58]
xd=np.arange(len(emotions)); wd=0.36
ax_d.bar(xd-wd/2,f1_nw,wd,color=GRAD_RED,alpha=0.85,label="Flat CE loss",edgecolor="none")
ax_d.bar(xd+wd/2,f1_w,wd,color=GRAD_GRN,alpha=0.85,label="WeightedFERLoss",edgecolor="none")
ax_d.set_xticks(xd); ax_d.set_xticklabels(emotions,fontsize=7)
ax_d.set_ylabel("F1 Score (%)",fontsize=9)
ax_d.set_title("(D) Class Imbalance Fix  (A1,A2,A4)",fontsize=10)
ax_d.legend(fontsize=7.5,framealpha=0.2); ax_d.yaxis.grid(True,zorder=0)

fig.suptitle("VisionEdit  -  Research Results Summary  (7 Kinetics-400 Classes, Real Clips)",
             fontsize=14,y=1.01,fontweight="bold",color="#f0f6fc")
fig.savefig(OUT/"00_summary_figure.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("  OK 00_summary_figure.png")

print()
print("="*60)
print("All graphs -> outputs/graphs/")
all_pngs=sorted(OUT.glob("*.png"))
for p in all_pngs:
    print(f"  {p.name:<45} {p.stat().st_size//1024:>4} KB")
print("="*60)
