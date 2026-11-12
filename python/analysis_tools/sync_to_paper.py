"""Copy the regenerated SI figures into the paper revision folder and rebuild the PDFs.
Paper folder: $QXCT_PAPER_DIR, else ../../../QuantumXCT_KL_paper/revision (repo layout)."""
import os, shutil, subprocess, glob
HERE = os.getcwd()
cands = [os.environ.get("QXCT_PAPER_DIR", ""), os.path.join(HERE, "..", "..", "..", "QuantumXCT_KL_paper", "revision")]
paper = next((d for d in cands if d and os.path.isdir(d)), None)
if paper is None:
    raise SystemExit("paper revision folder not found; set QXCT_PAPER_DIR")
for f in glob.glob("results/analyses/figures/*.pdf"):
    shutil.copy(f, paper)
# angle-initialization figure: prefer the sampled ovarian curve from run_angle_init,
# fall back to the legacy notebook output for backward compatibility.
ang_cands = (["results/analyses/figures/ovarian_angle_init_curves_sampled.png"]
             + sorted(glob.glob("results/analyses/figures/ovarian_angle_init_curves*.png"))
             + ["results/ovarian_cocultured/all_angle_init_curves.png"])
ang = next((p for p in ang_cands if os.path.exists(p)), None)
if ang:
    shutil.copy(ang, os.path.join(paper, "figS_angle_init.png"))
    print("angle-init figure:", ang)
else:
    print("WARNING: no angle-init figure found to copy")
print("figures copied to", os.path.abspath(paper))
if shutil.which("pdflatex"):
    for doc in ["sn-supplementary", "sn-article_revised", "response_to_reviewers"]:
        if not os.path.exists(os.path.join(paper, doc + ".tex")):
            continue
        for i in range(3):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", doc + ".tex"], cwd=paper, stdout=subprocess.DEVNULL)
            if i == 0 and doc != "response_to_reviewers" and shutil.which("bibtex"):
                subprocess.run(["bibtex", doc], cwd=paper, stdout=subprocess.DEVNULL)
        print("built", doc + ".pdf")
else:
    print("pdflatex not found: compile the .tex files in", paper)
