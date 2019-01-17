import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
def preprocess():
	authors_file = "SVM\email_authors.pkl"
	words_file = "SVM\word_data_unix.pkl"
	author = pd.read_pickle(authors_file)
	words = pd.read_pickle(words_file)
	
	features_train, features_test, labels_train, labels_test = train_test_split(words, author, test_size=0.1,
	                                                                            random_state=42)
	### text vectorization--go from strings to lists of numbers
	vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
	                             stop_words='english')
	features_train_transformed = vectorizer.fit_transform(features_train)
	features_test_transformed = vectorizer.transform(features_test)
	
	### feature selection, because text is super high dimensional and
	### can be really computationally chewy as a result
	selector = SelectPercentile(f_classif, percentile=1)
	selector.fit(features_train_transformed, labels_train)
	features_train_transformed = selector.transform(features_train_transformed).toarray()
	features_test_transformed = selector.transform(features_test_transformed).toarray()
	
	### info on the data
	print("no. of Chris training emails:", sum(labels_train))
	print("no. of Sara training emails:", len(labels_train) - sum(labels_train))
	
	return features_train_transformed, features_test_transformed, labels_train, labels_test

