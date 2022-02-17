from sklearn.neighbors import KNeighborsClassifier
domain = [{'name': 'n_estimators', 'type': 'discrete', 'domain':n_estimators},
          {'name': 'max_depth', 'type': 'discrete', 'domain': max_depth},
          {'name': 'max_features', 'type': 'categorical', 'domain': max_features},
          {'name': 'criterion', 'type': 'categorical', 'domain': criterion}]

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)