# Client_Feedback_Analysis
Analyzing whether Client feedback is Positive or Negative.

<h3> Description </h3>
We are using the reviews data from a Restaurant. All kinds of feedback  their customers provide and we are going to analyze whether those feedbacks are positive or negative feedbacks. So this dataset is only of two columns. One is feedback and other is type of feedback. We apply here some of Natural Language Processing (NLP) techniques to understand human feedback language.

<h3> Requirements </h3>
Hope you already have some python basics and you have ready any python environment like Jupyter. You need to install all the below libraries in your python environment. You can do this in Python IDLE using simple pip command.

     numpy, pandas, re, nltk, scikit-learn

<h3>1. Loading the Data</h3>
First we need to store the data file in your local machine somewhere and from that path we load the CSV file into Jupyter Notebook. For reading the file, we use PANDAS library. We mention the delimiter type as <tab> here.

     import pandas as pd
     dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

<h3>2. Cleaning the Data</h3>
We have the feedback in first 'Review' column. Now we have to clean this text. Cleaning involves of several steps. 
<br>
1. removing unwanted characters first. Like commans, punctuation marks any symbols and anything other than a-z letters. 
<br>
2. The second step is to remove stopwords. Stopwords are like a, the, is , was, yet, so... These kind of words are not useful while predicting customer feelings. So we remove them.
<br>
3. We have to divide the sentences into words. So we will have final words like good, bad, not, really, taste.. These words will help us understanding the client thoughts.
<br><br>
So we do all these steps and put the final words in a python array. Here we use the famous 'nltk' library which is widely used on this process. It already has the all stopwords in 'stopwords' module, for several languages.
<br>
So let's see the code
<br>

     import re
     import nltk
     nltk.download('stopwords')
     from nltk.corpus import stopwords
     from nltk.stem.porter import PorterStemmer
     corpus = []
     for i in range(0, 1000):
         review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
         review = review.lower()
         review = review.split()
         ps = PorterStemmer()
         review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
         review = ' '.join(review)
         corpus.append(review)
         
Now in corpus array we have all the important words we need.
<br>
<h3>3. Creating the Bag of Words model</h3>
Now this step is very important. The question is how do we use all these words in a machine learning algorithm. It has many lines of arrays of words, we have to parse or transform them to train with a model.
<br> Lets see a small example how it has to be done, before going into the code. If we have 2 reviews as 'good taste' and 'bad taste'. (Customer may not give exactly like this, they might have said 'This is Good in Taste', but we removed stop words and converted them to lower case.) So we make those words are features and replace them what are words are given by review1 and review2 and so on. Please see below.

<img src=screenshots/transform_example.jpg width=40% />
 
 For all the rows, we might see large number of words, we can limit these numbers in the code. Let's see code.
 
      from sklearn.feature_extraction.text import CountVectorizer
      cv = CountVectorizer(max_features = 1500)
      X = cv.fit_transform(corpus).toarray()
      y = dataset.iloc[:, 1].values
 
<h3>4. Splitting Data</h3>
We also seperated the output column and input column. We can also say Independent columns(input) and Dependent column(output). Here our output variable is the last column income. The next step is splitting. We use 80 to 90 % of the data for training the model and we test it on the remaining 10% of the data which is called test data. This split can be randomly applied by the Python function 'train_test_split'.
 
      # Splitting the dataset into the Training set and Test set
     from sklearn.cross_validation import train_test_split
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

<h3>5. Training the Data</h3>
We are applying the Naivebayes model first to the data to do probabilistic analysis. Because of independent variable is binary, so it is like finding the combination of words or finding the words which actually leading to good feedback.

     from sklearn.naive_bayes import GaussianNB
     classifier = GaussianNB()
     classifier.fit(X_train, y_train)

<h3>6. Evaluating the model</h3>
we calculate y_pred, the predicted outcome on the test data we kept aside. We have to compare this y_pred with y_test values and check how much data is a match. This is how we estimate the accuracy of the model. We have certain measure like 'accuracy_score' , 'confusion_matrix' and 'classification_report' to evaluate the model. We can use 'metrics' library in python to find these values.

     # Predicting the Test set results
     y_pred = classifier.predict(X_test)

     # Making the Confusion Matrix
     from sklearn.metrics import confusion_matrix
     cm = confusion_matrix(y_test, y_pred)
     print(metrics.accuracy_score(y_test, y_pred);
     
     Out: 0.8597005208833334

<br>
Lets see the model evaluation with an example.
Consider a customer said, "The food quality is very very bad had order some soup it was so terrible could eat more than a spoonful. They need to change the chef at the earliest." For this we apply same logics as above and see what will be the outcome.

     text = "The food quality is very very bad had order some soup it was so terrible could eat more than a spoonful. They need to change the chef at the earliest."
     corpus2 = []
     review2 = re.sub("[^a-zA-Z]", ' ', text)
     review2 = review2.lower()
     review2 = review2.split()
     ps2 = PorterStemmer()
     review2 = [ps2.stem(word) for word in review2 if not word in set(stopwords.words('english'))]
     review2 = " ".join(review2)
     corpus2.append(review2)
     from sklearn.feature_extraction.text import CountVectorizer
     cv2 = CountVectorizer(max_features = 1500)
     X2 = cv2.fit_transform(corpus + corpus2).toarray()
     my = X2[-1].reshape(1, -1)
     result = classifier.predict(my)
     if result == 1:
        answear = "Positive"
     else:
        answear = "Negative"
    
     print(answear)
     
     Out: Negative

<h3>7. Performace Tuning using TF-IDF </h3>
In the world of NLP, we mostly deal with the text which is given/generated by human. Well humans can give information in their own way of understanding and so there are some hidden problems we face when we convert this text data into a training data set for a machine learning algorithm. We mainly observe 2 kinds of problems below.
<br><br>
1. Let's consider for our Restaurant, we mainly get reviews in three categories - Starters, Main course and Desserts. Customers while giving feedback for Starters, they might keep mentioning the word 'starter'. With this the frequency of this word (Term Frequency) will be high for this category, same with other categories. This unnecessary high frequency suppresses the value of other important words that actually need to be used in machin learning training process.
<br>
2. Next the other problem can with the words like 'food', which will present in all the three categories of starters, main course as well as desserts. If there is a common word with good frequency in all the categories (Inverse Document Frequency), the model unable to train/classify whether a feedback is related to which category in the first place.
<br><br>
To solve these problems we use the concept called TF-IDF (Term Frequency - Inverse Document Frequency). The purpose of TF_IDF is to highlight the words which are frequent in a category but not across categories. The basic idea is we divide the frequency count with the total number of words in each category, this will normalise the all the counts and the problem with high frequency of words will be reduced.
<br><br>
In this process, we use a formula to multiply the frequency of a word in particular category is by a logarithmic function which divides the total number of words with the frquency of that word in number of categories. Below formula will help in supressing the effect of commonly occuring words in all categories. This is very useful in the text analytics.
<br><br>

     w(i,j) = f(i,j)*log(N/f(i))
     
     Here w(i,j) = for a Word i in j category
          f(i,j) = frequency of word i in j category
          N      = Number of categories
          f(i)   = Frequency of word i occuring in number of categories
 
<br>
In python, we can use 'feature_extraction' library in sklearn to achieve this. Please refer below code. A unique bag of words from entire dataset is referred as a corpus in general terms.
<br><br>

     from sklearn.feature_extraction.text import TfidfVectorizer
     vectorizer = TfidfVectorizer()
     vectorizer.fit(corpus)

<br>
<h3>Conclusion</h3>
Congratulations, you made it this far. We have read, cleaned and processed the data. We trained data with naive bayes model and got the results. From the above we see the model is giving good accuracy score 85.9%. This can be increased after TF_IDF conversion and as well as any Bagging or Boosting techniques, by means training multiple times by shuffling, different sets of data.
<br><br>
Thank you..
