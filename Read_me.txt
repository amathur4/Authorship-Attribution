1. Our code is in Python programming language (Python 3.6) and we used Anaconda Navigator- Spyder (version 3.3.2) IDE.
2. The code that we imlemented makes use of methods from textstat library. In order to be able to run our code please type the following command in Anaconda prompt to install textstat.
   'pip install textstat'.
3. Average word frequency class code is in a jupyter notebook file (version 5.7.4). The output of that is a text file titled 'avg_wrd_freq.txt'.
4. The text file '1_1000.txt' represents the 1000 authors that we considered.
5. The text file '1_500.txt' represents the first set of 500 authors that we considered.
6. The text file '501_1000.txt' represents the second set of 500 authors that we considered.
7. The files avg_wrd_freq.txt, 1_1000.txt, 1_500.txt, 501_1000.txt should be in the same folder as the python files.
8. We used MySQL to store the dataset. 
	The data that we used is provided in http://www2.cs.uh.edu/~arjun/courses/ml/Projects.pdf in Authorship attribution slide (Review data set used in the baseline paper, the '.7z' compressed file)
	To get the data :
	8.1 Install MySQL (Community Edition) https://www.mysql.com/downloads/
	8.2 Paste the sql file inside the bin folder of mysql server folder.
	8.3 Open the MySQL command line.
	8.4 Type 'source amazon_software.sql' (source filename.sql) and enter.
	8.5 Sql file will be uploaded successfully.
9. A connection to the MySQL is required to obtain the data. Please download MySQL Connector/Python(https://dev.mysql.com/downloads/connector/python/)
10. Open the .py file in spyder. Replace the credentails provided here with your username and password in the python file (.py file). 
  [connection = sql.connect(host='localhost', database='amazon_db_software', user='Aparna', password='anu90appu96@')]
11. Press the 'run' icon (the green arrow pointing to the right).
12. To do a comparision between the effects of the number of authors on performance we also have 1_200.txt representing 200 authors, 1_400.txt representing 400 authors, 1_600.txt representing 600 authors, 1_800.txt representing 800 authors.
    All these text files should be in the same folder as the python file (.py file)
13. The results shown in the presentations were not obtained by doing cross-validation whereas the ones in the report are obatined by doing 5-fold cross-validation.
14. ML_CODE folder contains the .py and .txt files required to run the code.
	14.1 ml_adaboost_allfeatures.py is the code for adaboosting
	14.2 ml_dt_allfeatures.py contains the code for Decision tree classifier
	14.3 ml_logistic_regression_allfeatures.py contains the code for Decision tree classifier
	14.4 ml_nb_allfeatures.py contains the code for Naive Bayes classifier
	14.5 ml_neural_MLP_allfeatures.py contains the code for MLP
	14.6 ml_svm_allfeatures.py contains the code for Linear svm classifier
	14.7 ml_two_sets_of_authors.py contains the code where the 1000 authors are divided into two sets of 500 each
	14.8 ml_featureselection.py contains the code for chi square feature selection
	14.9 ml_featureimp_graph_1author.py contains the code to visualize feature importance graph for 1 author
	14.10 average word frequency class folder contains the jupyter notebook code along with 2 .rar files, pan12.rar and pan13.rar(contains PAN-13 dataset along with 10 e-books)
