# SpeechImpactPrediction
Predicting Views, TalkType and Popularity of Speeches

Speeches play a pivotal role in inspiring and galvanising individuals and have the power to reach the masses. The topic, content, and delivery style directly contribute to the popularity or the number of views for the speech. That is why it is valuable to predict these factors to understand how speeches resonate with the audience.

From human judgement, the subject matter of the talks or the speaker themselves are often seen to be the driving factor behind their acclaim. This project focuses on implementing a textual analyzer that combines other numerical facts of the data to learn the fundamental themes included in a formal address. The solution we propose utilizes concepts across Natural Language Processing and other data mining techniques to analyze transcripts of TED talks. Many popular approaches are explored across the data mining pipeline to answer the following research questions:

- RQ1: Can we predict view counts of a talk (Regression)?
- RQ2: Can we predict the type of a talk(Classification)?
- RQ3: Can we predict the popularity of a talk (Classification)?

# Running the script
Make sure the .csv files are inside a folder directory named "data", and the "temp" and "result" directories also exist. Also make sure the .py files are in the same directory. The directory structure in which you run the code should be similar to this repository.
Simply run "python TedEx.py", and it will output all the results and also saves the results files in the result directory.
The report for this project can be found in the "docs" folder.

# Conclusion
In the order of the questions posed above, we summarize our results:

- RQ1: The view count for the speech could be extrapolated to some extent.  The best performing regression model over the test data was ridge regression.  This may imply that overfitting to the training data cost the model accuracy. 

- RQ2: We observed that the Support Vector Classifier with an arbitrary penalty factor of 100 and a linear kernel seems to outperform the other two classifiers we considered for the talkType label with an F1-score of 0.5269. Tuning the hyperparameters of the Support Vector Classifier using FLASH seems to have boosted its performance even more giving an F1-score of 0.569, an increase of 6.8%

- RQ3: For the classification task of talkType target, we observe the Support Vector Classifier again outperforming  the  other  models  with  an  F1-score  of  0.465.   The  performance  is  boosted  with hyperparameter tuning, which yields an F1-score of 0.5245, an improvment of 12.7%.
