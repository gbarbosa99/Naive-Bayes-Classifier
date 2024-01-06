from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'])
print(emails.target_names)

# emails are stored in emails.data
print(emails.data[5])

# labels are stored in emails.target
print(emails.target[5])

# labels themselves are #s but they correspond to label names found at emails.target_names
print(emails.target_names[1])
# target of email 5 is 1, which corresponds to rec.sport.hockey

# split data into training and test sets
train_emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'], subset = 'train', shuffle = True, random_state = 108)
test_emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'], subset = 'test', shuffle = True, random_state = 108)

# Create a counter object to transform emails into lists of word counts
counter = CountVectorizer()
counter.fit(test_emails.data + train_emails.data)

# Make a list of the counts of our words in the training set
train_counts = counter.transform(train_emails.data)
test_counts = counter.transform(test_emails.data)

# Create a naive bayes classifier object to train and test on
classifier = MultinomialNB()

classifier.fit(train_counts, train_emails.target)

# .score() returns the accuracy of the classifier on the test data measuring the percentage of classifications a classifier correctly makes
print(classifier.score(test_counts, test_emails.target))
# It appears the classifier does a good jov distinguishing between soccer and hockey emails

