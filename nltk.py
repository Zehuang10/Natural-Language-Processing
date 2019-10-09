## NLTK
import nltk

#nltk.download_shell()

messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]

print(len(messages))

messages[0]

for mess_no, message in enumerate(messages[:10]):
    print(mess_no,message)
    print('\n')

messages[0]

import pandas as pd

Read to separate the \t 

messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',names=['label','message'])

messages.head()

messages.describe()

messages.groupby('label').describe()

messages['length'] = messages['message'].apply(len)

messages.head()

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.set_style('darkgrid')
messages['length'].plot.hist(bins=30)

messages['length'].plot.hist(bins=110)

messages.describe()

messages[messages['length']==910]['message'].iloc[0]

messages.hist(column='length',by='label',bins=60,figsize=(12,5))

import string

mess = 'Sample Message! Notice: It Has punctuation.'

string.punctuation

no_punc = [c for c in mess if c not in string.punctuation]

no_punc

Now remove stopwords that do not add value

from nltk.corpus import stopwords

stopwords.words('english')

Join elements in a list together with .join()

no_punc = ''.join(no_punc)

no_punc

x = ['a','b','c','d']

x

''.join(x)

no_punc.split()

clean_mess = [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]

clean_mess

def text_process(mess):
    """
    1. remove punctuation
    2. remove stopwords
    3. return list of clean words 
    """
    
    nopunc = [char for char in mess if char not in string.punctuation]
    
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
text_process(messages.iloc[0][1])

messages.loc[0][1]

messages.head()

messages['message'].head().apply(text_process)

Create a sparse matrix to store word counts for different messages

from sklearn.feature_extraction.text import CountVectorizer

bag_of_words_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

Check how many words we are counting

print(len(bag_of_words_transformer.vocabulary_))

sample = messages['message'][3]

bow_sample = bag_of_words_transformer.transform([sample])

We can see that we have 7 words with 2 words repeating twice 

print(bow)

print(sample)

bag_of_words_transformer.get_feature_names()[4068]

bag_of_words_transformer.get_feature_names()[9554]

messages_bow = bag_of_words_transformer.transform(messages['message'])

print('Shape of the Sparse Matrix: ', messages_bow.shape)

# Grab non zero occurences in our example
messages_bow.nnz

sparsity = (100 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(sparsity))

Now we can analyze term frequency of the words

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)

tfidf_sample = tfidf_transformer.transform(bow_sample)

print(tfidf_sample)


tfidf_transformer.idf_[bag_of_words_transformer.vocabulary_['university']]

messages_tfidf = tfidf_transformer.transform(messages_bow)

We will now use Naive Bayes to train data

from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB().fit(messages_tfidf,messages['label'])

spam_detect_model.predict(tfidf_sample)[0]

all_predictions = spam_detect_model.predict(messages_tfidf)

all_predictions

Now test model on our test data to see accuracy of our model

from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'],test_size=0.3)

## We can create a pipeline with all the steps we previously made to facilitate our work

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bag_of_words_transformer', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(msg_train, label_train)

predictions = pipeline.predict(msg_test)

from sklearn.metrics import classification_report

print(classification_report(label_test, predictions))

Using Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('bag_of_words_transformer', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

pipeline.fit(msg_train, label_train)
predictions = pipeline.predict(msg_test)
print(classification_report(label_test, predictions))
