# geoproject

Step 1 : $ pip install -r requirements.txt

Step 2 : look at code_rndxCNN.ipynb. You can tweak the features and run the cells.

Step 3 : your features and results are saved in ./results/automatic_study_features_rndxCNN.xlsx. If you add a feature, a new column will be created. If you remove one, the value in the column will be None.
Take care to close the excel file before running the files and to install the requirements (there is some weird librairies to install in order to handle xlsx files in pandas)


Comments/TODO (old):

- Do the CNN features induce a wrong estimation of the cross validation f1 score as the CNN has probably train on some samples used in the validation set ?
=> the higher the k_fold_number, the higher the standard deviation of the cross validation f1 score. So I tend to say that it does matter


- How do we choose the best features among all these ? Will dimensionality reduction/PCA improve the Random Forest predictions ?

- What other features can we craft ? (change urban_type in MultiBinaryEncoder (one-hot))

- How does the `k_fold_number` change the cross validation f1 score ? [I guess that the higher is k, the larger is the training set and the closer is the Random Forest estimated to the Random Forest whose predictions are submitted]

- When to submit ? 





TODO 27/01:


- proximité data test entrainment : 
est-ce que les données de test sont vraiment très loin des donné s’il y a une disparité, il faut peut-être penser à faire plus d’effort sur ces différences (sans perdre en généralisation). Exemple : s’il y a beaucoup de NaN dans les données de test, on peut réfléchir à une meilleure gestion de ces NaN.


- Paul T. : classes problématiques (comme Mega Project) : 
on peut penser à une stratégie à part pour eux du style utiliser des stratégies de détection d’anomalie plutôt que de la classification classique.
regarder les données pour voir s’il y a qqc que nous avons oubliés dedans


hyperparamètres xgboost :
optuna plz


réduction dimension non supervisée : 
regarder le cours sur le sujet : https://centralesupelec.edunao.com/pluginfile.php/298226/mod_resource/content/1/2EL1730-ML-Lecture09-Dimensionality_Reduction%20-%20NO.pdf
Greedy wrapper approach 


Tj de nouvelles features : 
créer un vecteur directeur moyen pour prendre en compte les séquences à la place du NN
Les valeurs donner par encoding sont-elles bien réparties ? Est-ce que on n’a pas l’équivalent de {0:”en construction”,1:”destruction”,2:”construction terminée”} ? Si c’est le cas, le NN et le second modèle va galérer à trouver des séparateurs ou interpoler.


JR : Avoir une super k-fold cross-validation qui est identique sur le NN et le Random Forest = au moins on est sûr que la validation est faite sur des données vues par personne
=> finalement, très mauvaise idée : on aurait k model NN et k model de random forest puisque les features à choisir serait différentes pour chaque + quand on passe à l'entraînement sur l'ensemble du dataset, on devrait avoir encore un autre model de toute façon.
=> solution : train le NN sur l'ensemble de dataset d'entraînement. Puisqu'on veut les features, ça ne devrait pas être trop grave niveau mesure de la capacité à généraliser.

