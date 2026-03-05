import os, h5py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

sns.set(style="whitegrid", context="talk")

ROOT = "/mnt/deepstore/Vidur/Junk Classification/data_model_labels"

def count_hdf5_rows(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".hdf5")])
    counts = {}
    for f in files:
        path = os.path.join(folder, f)
        try:
            with h5py.File(path, "r") as hf:
                if "images" in hf:
                    n = hf["images"].shape[0]
                else:
                    # fallback: count datasets of equal length
                    n = list(hf.values())[0].shape[0]
                counts[f] = n
        except Exception as e:
            print(f"Error reading {f}: {e}")
    return pd.DataFrame(list(counts.items()), columns=["filename", "n_cells"])

folders = ["junk_annotated", "rare_cells_annotated", "wbcs_annotated", "unannotated"]
summary = {}
for fld in folders:
    path = os.path.join(ROOT, fld)
    if os.path.exists(path):
        df = count_hdf5_rows(path)
        summary[fld] = df

# quick preview
for k, v in summary.items():
    print(f"\n{k}: {v['n_cells'].sum()} total cells")
    display(v)

total_counts = {k: v['n_cells'].sum() for k,v in summary.items() if k != "unannotated"}
df_total = pd.DataFrame(list(total_counts.items()), columns=["Category", "Count"])
plt.figure(figsize=(6,5))
sns.barplot(data=df_total, x="Category", y="Count", palette="muted")
plt.title("Annotated Dataset Composition")
plt.ylabel("Number of Cells")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("dataset_composition_bar.png", dpi=200)
plt.show()

rare_df = summary["rare_cells_annotated"].copy()
rare_df["Subtype"] = rare_df["filename"].str.replace(".hdf5","",regex=False)
plt.figure(figsize=(8,5))
sns.barplot(data=rare_df, x="Subtype", y="n_cells", palette="viridis")
plt.title("Rare Cell Subtype Distribution")
plt.ylabel("Number of Cells")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("rare_subtypes_bar.png", dpi=200)
plt.show()

mm_df = summary["unannotated"].copy()
mm_df["Cluster"] = mm_df["filename"].str.extract(r'(\d+)').astype(int)
mm_df = mm_df.sort_values("Cluster")

plt.figure(figsize=(8,5))
sns.barplot(data=mm_df, x="Cluster", y="n_cells", palette="rocket")
plt.title("MM Cluster Cell Counts")
plt.ylabel("Number of Events")
plt.xlabel("MM Cluster ID")
plt.tight_layout()
plt.savefig("mm_cluster_counts_bar.png", dpi=200)
plt.show()

# optional pie chart
plt.figure(figsize=(6,6))
plt.pie(mm_df["n_cells"], labels=mm_df["Cluster"], autopct="%1.1f%%",
        colors=sns.color_palette("rocket", len(mm_df)))
plt.title("Proportion of Cells per MM Cluster")
plt.tight_layout()
plt.savefig("mm_cluster_counts_pie.png", dpi=200)
plt.show()

summary_table = []
for cat, df in summary.items():
    for _, row in df.iterrows():
        summary_table.append({
            "Category": cat,
            "Filename": row["filename"],
            "Count": row["n_cells"]
        })
pd.DataFrame(summary_table).to_csv("dataset_summary_counts.csv", index=False)
print("Saved summary table: dataset_summary_counts.csv")