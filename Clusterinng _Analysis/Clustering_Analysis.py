import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("customer_churn_preprocessed_Data.csv")

# print(df.head())
X = df.copy()
pca = PCA()

X_pca = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_

# print(explained_variance)
plt.figure(figsize=(8,5))
plt.plot(
    range(1, len(explained_variance)+1),
    pca.explained_variance_ratio_.cumsum(),
    marker='o'
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")

plt.grid()
plt.show()

#Using PCA for Dimensionality reduction

pca = PCA(n_components=8)
X_reduced = pca.fit_transform(X)
print(X_reduced.shape)


#Using elbow method to find the optimal number of clusters

wcss = []
for k in range(2,11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_reduced)

    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(
    range(2,11),
    wcss,
    marker="o"
)
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.title(
    "Elbow Method After PCA to get optimal number of clusters"
)
plt.show()


#calculating Silhoutte_Scores for determining number of clusters.

silhouette_scores = []
for k in range(2,11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    labels = kmeans.fit_predict(X_reduced)
    score = silhouette_score(
        X_reduced,
        labels
    )

    silhouette_scores.append(score)
    
plt.figure(figsize=(8,5))
plt.plot(
    range(2,11),
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title(
    "Silhouette Scores"
)
print(silhouette_scores)
plt.show()


#Selecting optimal number of clusters, Training the k - means model  and visualizing using sactterplot

optimal_clusters = 5
kmeans = KMeans(
    n_clusters=optimal_clusters,
    random_state=42,
    n_init=10
)
clusters = kmeans.fit_predict(X_reduced)
df["Cluster"] = clusters
df["Cluster"].value_counts()
pca_visual = PCA(n_components=2)
X_visual = pca_visual.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(
    X_visual[:,0],
    X_visual[:,1],
    c=clusters
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title(
    "Customer Segmentation Using PCA + K-Means"
)
plt.show()


#Tnterpreting the clusters/ profiling. 

cluster_profile = (
    df.drop(columns=["Churn"])
      .groupby("Cluster")
      .mean()
)

cluster_profile = df.groupby("Cluster").mean()

#----------churn rate per cluster---------------------
churn_rate = df.groupby("Cluster")["Churn"].mean()

#----------Cluster size -------------------------------
cluster_size = df["Cluster"].value_counts().sort_index()

#----------Combining everything--------------------
summary = cluster_profile.copy()
summary["ClusterSize"] = cluster_size
summary["ChurnRate"] = churn_rate

#---------Converting churn rate to percentage--------------
summary["ChurnRate"] = (summary["ChurnRate"] * 100).round(2)

#--------Rounding all values for readability --------------------
summary = summary.round(3)
print(summary)
summary = summary[["ClusterSize", "ChurnRate"] +
                  [col for col in summary.columns
                   if col not in ["ClusterSize", "ChurnRate"]]]

print(summary)
summary_pct = summary.copy()

# Convert selected binary columns to percentages
binary_cols = ["SeniorCitizen", "Contract_One year", "Contract_Two year", "Contract_Month-to-month"]

for col in binary_cols:
    summary_pct[col] = (summary_pct[col] * 100).round(1)

print(summary_pct)


#Adding a risk Label on the profile table 
def risk_label(rate):
    if rate >= 40:
        return "High"
    elif rate >= 20:
        return "Medium"
    elif rate >= 10:
        return "Low"
    else:
        return "Very Low"

summary_pct["RiskLevel"] = summary_pct["ChurnRate"].apply(risk_label)

print(summary_pct[["ClusterSize", "ChurnRate", "RiskLevel",
                   "SeniorCitizen", "Contract_One year", "Contract_Two year", "Contract_Month-to-month"]])

#creating visualisation table :

cluster_profile = df.groupby("Cluster").mean()
churn_rate = df.groupby("Cluster")["Churn"].mean()
cluster_size = df["Cluster"].value_counts().sort_index()

summary = pd.DataFrame({
    "Cluster Size": cluster_size,
    "Churn Rate (%)": (churn_rate * 100).round(1),
    "Senior Citizens (%)": (cluster_profile["SeniorCitizen"] * 100).round(1),
    "One-Year Contract (%)": (cluster_profile["Contract_One year"] * 100).round(1),
    "Two-Year Contract (%)": (cluster_profile["Contract_Two year"] * 100).round(1),
    "Month-to-month Contract (%)": (cluster_profile["Contract_Month-to-month"] * 100).round(1)

})

# Add risk label
summary["Risk Level"] = pd.cut(
    summary["Churn Rate (%)"],
    bins=[0, 10, 20, 40, 100],
    labels=["Very Low", "Low", "Medium", "High"],
    


 include_lowest=True
)


# Sort clusters by highest churn rate first
summary = summary.sort_values(
    by="Churn Rate (%)",
    ascending=False
)

# Display styled table
styled_summary = (
    summary.style
    .background_gradient(subset=["Churn Rate (%)"], cmap="Reds")
    .background_gradient(subset=["Senior Citizens (%)"], cmap="Blues")
    .set_caption("Customer Segmentation Summary (PCA + K-Means)")
    .format({
        "Churn Rate (%)": "{:.1f}",
        "Senior Citizens (%)": "{:.1f}",
        "One-Year Contract (%)": "{:.1f}",
        "Two-Year Contract (%)": "{:.1f}",
        "Month-to-month Contract (%)": "{:.1f}"
 
    })
)

styled_summary

html = styled_summary.to_html()
with open("cluster_summary.html", "w") as f:
    f.write(html)
# print("Table saved as cluster_summary2.html")