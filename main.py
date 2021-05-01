from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True)
naive_bayes=MultinomialNB()

ball_train = pd.read_csv('train/train.tsv', sep='\t', error_bad_lines=False, header=None)
print(ball_train.head())
print(len(ball_train))
print(pd.DataFrame(ball_train[0]))
print(pd.DataFrame(ball_train[1]))

y_train = pd.DataFrame(ball_train[0])
x_train = pd.DataFrame(ball_train[1])
x_train.cv=cv.transform(x_train)

naive_bayes.fit(x_train.cv, y_train)

ball_dev = pd.read_csv('dev-0/in.tsv', sep='\t', error_bad_lines=False, header=None)
with open('dev-0/expected.tsv', 'r') as dev_exp_f:
    Y_dev = np.array([float(x.rstrip('\n')) for x in dev_exp_f.readlines()])


X_dev = pd.DataFrame(ball_dev)
X_dev.cv=cv.transform(X_dev)

Y_dev_predicted = naive_bayes.predict(X_dev.cv)
pd.DataFrame(Y_dev_predicted).to_csv('dev-0/out.tsv', sep='\t', index=False, header=False)


ball_test=pd.read_csv('test-A/in.tsv', sep='\t', error_bad_lines=False, header=None)
X_test = pd.DataFrame(ball_test)
X_test.cv=cv.transform(X_test)

Y_test_predicted = naive_bayes.predict(X_test.cv)
pd.DataFrame(Y_test_predicted).to_csv('test-A/out.tsv', sep='\t', index=False, header=False)