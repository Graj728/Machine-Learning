#Project 7: Clustering

# Imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import cdist
#---------- Dataset ----------
# Six points A-F with X and Y coordinates, stored in a pandas DataFrame indexed by point name
d = {"point": list("ABCDEF"), "X": [1, 2, 3, 8, 9, 10], "Y": [2, 3, 4, 7, 8, 9]}
df = pd.DataFrame(data=d).set_index("point")
X = df[["X", "Y"]].values #Array of coordinates
names = list(df.index)
 
 
# Divisive clustering using the distance based splitting
D = cdist(X, X)
print("Distance Matrix:\n", pd.DataFrame(D.round(2), index=names, columns=names), "\n")
 
#method to convert centroid into list
def tolist(c):
    return [round(float(v), 2) for v in c]
 
 
log = [] #table for steps
 
# step1: one cluster and computing centroid
center_all = X.mean(axis=0)
log.append(("Step 1 ", "All points is in one cluster and computing centroid", "ABCDEF", f"{tolist(center_all)}"))
 
# step2: two points far from the each other become split
i, j = np.unravel_index(np.argmax(D), D.shape)  # largest distance
i, j = min(i, j), max(i, j)

log.append(("Step 2", f"Farthest pair = {names[i]}, {names[j]} (distance {D[i, j]:.2f}); use as seeds",
            f"Seed 1: {names[i]} | Seed 2: {names[j]}", f"{tolist(X[i])}; {tolist(X[j])}"))
 
# Step 3: assign every point to the nearer seed
centers = np.array([X[i], X[j]])
labels = cdist(X, centers).argmin(axis=1)
grp = lambda k: "".join(n for n, l in zip(names, labels) if l == k)
log.append(("Step 3", "Assign each point to the nearer seed",
            f"C1: {grp(0)} | C2: {grp(1)}", "-"))
 
# Step 4: recompute centroids, then re-assign by distance to centroids until stable
step = 4
while True:
    centers = np.array([X[labels == k].mean(axis=0) for k in (0, 1)])
    log.append((f"Step {step}", "Recompute centroids", f"C1: {grp(0)} | C2: {grp(1)}",
                f"C1 {tolist(centers[0])}; C2 {tolist(centers[1])}"))
    new_labels = cdist(X, centers).argmin(axis=1)
    step += 1
    if (new_labels == labels).all():
        log.append((f"Step {step}", "Re-assign by distance to centroids: no point changes -> stop (2 clusters)",
                    f"C1: {grp(0)} | C2: {grp(1)}", f"C1 {tolist(centers[0])}; C2 {tolist(centers[1])}"))
        break
    labels = new_labels
    step += 1
#table for steps 
dc_table = pd.DataFrame(log, columns=["Step", "Action", "Clusters", "Centroids / Seeds"])
pd.set_option("display.width", 250, "display.max_colwidth", 120)
print("\nDivisive clustering steps:\n", dc_table.to_string(index=False), "\n")

dc_labels = labels.copy()

# K-M
km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X)
df["KMeans"] = km.labels_
print("k-means centroids:\n", km.cluster_centers_.round(2))
 
# Agglomerative (Ward linkage, 2 clusters)
agg = AgglomerativeClustering(n_clusters=2, linkage="ward").fit(X)
df["Agglomerative"] = agg.labels_
print("\n", df, "\n")
 
# Plots
colors = np.array(["tab:blue", "tab:red"])
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for a, labels, title in zip(ax, [dc_labels, km.labels_, agg.labels_],
                            ["Divisive (distance-based)", "k-means (k=2)", "Agglomerative (Ward, k=2)"]):
    a.scatter(X[:, 0], X[:, 1], c=colors[labels], s=90)
    for i, n in enumerate(names):
        a.annotate(n, X[i], textcoords="offset points", xytext=(6, 6))
    for c in np.unique(labels):
        cen = X[labels == c].mean(axis=0)
        a.scatter(*cen, marker="X", s=200, c="black")
    a.set_title(title)
    a.set_xlabel("X")
    a.set_ylabel("Y")
    a.grid(alpha=.3)
plt.tight_layout()
plt.show()
 
Z = linkage(X, method="ward")
plt.figure(figsize=(6, 4.5))
dendrogram(Z, labels=names)
plt.title("Agglomerative clustering dendrogram (Ward)")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()