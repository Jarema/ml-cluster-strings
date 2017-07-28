import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import extract_features as ef
from sklearn.cluster import KMeans

dataset = np.genfromtxt('dataset.csv', delimiter=',')
(m,n)   = dataset.shape
X = dataset[:, 0:n-1]

np.random.seed(200)

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
kmeans.labels_


manual_test = (
	'IYlkjHJKglh',
	'PZU',
	'PZU S.A.',
	'JanKowalski',
	'Jan Kowalski',
	'JanKo Walski',
	'LjwNoFOMYAivbZBoP',
	'Bg.pEGQBzXnJMTvbZB',
	'RwvqLUvaJMbIvbBoP',
	'QGjsqvbZBoPervJjq',
	'ovqd0bZB oPovbZBoP',
	'yvbZB2oPńvbZBoPG',
	'vbZBoP333IABiJzkf',
	'JzHpnTvbZBoPi3oJI',
	'xvehllzvbz555bpwz',
	'FZXHEVYLHJLGCCNPN',
	'dupa',
	'DUPA',
	'Dupa',
	'DuPa'
)
for example in manual_test:
	prediction = kmeans.predict(ef.extractFeatures(example).reshape(1, -1))
	if prediction[0]==1:
		print('Example {0}\t\tis considered a hash'.format(example))
	else:
		print('Example {0}\t\tis considered a plain text'.format(example))
