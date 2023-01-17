# geoproject

Step 1 : $ pip install -r requirements.txt

Step 2 : look at code_rndxCNN.ipynb. You can tweak the features and run the cells.

Step 3 : your features and results are saved in ./results/automatic_study_features_rndxCNN.xlsx. If you add a feature, a new column will be created. If you remove one, the value in the column will be None.
Take care to close the excel file before running the files and to install the requirements (there is some weird librairies to install in order to handle xlsx files in pandas)


Comments/TODO:

- Do the CNN features induce a wrong estimation of the cross validation f1 score as the CNN has probably train on some samples used in the validation set ?
=> the higher the k_fold_number, the higher the standard deviation of the cross validation f1 score. So I tend to say that it does matter


- How do we choose the best features among all these ? Will dimensionality reduction/PCA improve the Random Forest predictions ?

- What other features can we craft ? (change urban_type in MultiBinaryEncoder (one-hot))

- How does the `k_fold_number` change the cross validation f1 score ? [I guess that the higher is k, the larger is the training set and the closer is the Random Forest estimated to the Random Forest whose predictions are submitted]

- When to submit ? 