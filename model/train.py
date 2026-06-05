from sklearn.cluster import AgglomerativeClustering

def train_model(X, n_clusters):

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward"
    )

    clusters = model.fit_predict(X)

    return model, clusters