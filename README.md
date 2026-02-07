# Classification des types de contrat – Random Forest

## 1️⃣ Objectif

Ce projet a pour objectif de **prédire le type de contrat (`CDD`, `CDI`, `Indépendant`) des employés** en fonction de leurs caractéristiques socio-économiques et professionnelles (âge, revenu mensuel, ancienneté, dépense annuelle, secteur, région).  
Le notebook illustre une démarche complète de **prétraitement, exploration et modélisation supervisée** avec un pipeline réutilisable pour d’autres jeux de données similaires.

---

## 2️⃣ Préparation et exploration des données

1. **Chargement et nettoyage**
   - Suppression des doublons  
   - Imputation des valeurs manquantes : moyenne pour les variables numériques, mode pour les variables catégorielles

2. **Exploration**
   - Statistiques descriptives des variables numériques  
   - Visualisation des distributions et relations entre variables  
   - Scatter plots entre chaque variable numérique et la variable cible `type_contrat` pour observer les tendances

---

## 3️⃣ Prétraitement et split

- Les variables numériques et catégorielles sont traitées via un **ColumnTransformer** :
  - StandardScaler sur certaines variables numériques  
  - MinMaxScaler sur d’autres variables numériques  
  - One-Hot Encoding sur les variables catégorielles
- Séparation du dataset en **train/test (80/20)**

---

## 4️⃣ Modèle et pipeline

- Un **pipeline** combine prétraitement et modèle :  
  - Prétraitement (`preprocessor`)  
  - **RandomForestClassifier** avec 100 arbres et `random_state=42`  
- Entraînement sur le **train set** : `pipeline.fit(X_train, y_train)`

---

## 5️⃣ Prédiction et évaluation

- Prédiction sur le **test set** : `y_pred = pipeline.predict(X_test)`  
- **Accuracy obtenue** : 0.367
- **Confusion Matrix** :
  
| True \ Pred | CDD | CDI | Indépendant |
|------------|-----|-----|-------------|
| CDD        | 1   | 14  | 2           |
| CDI        | 8   | 20  | 4           |
| Indépendant| 2   | 8   | 1           |


- **Classification Report** :

| Classe         | Precision | Recall | F1-score | Support |
|----------------|-----------|--------|----------|--------|
| CDD            | 0.09      | 0.06   | 0.07     | 17     |
| CDI            | 0.48      | 0.62   | 0.54     | 32     |
| Indépendant    | 0.14      | 0.09   | 0.11     | 11     |
| **Accuracy**   |           |        | 0.37     | 60     |
| **Macro avg**  | 0.24      | 0.26   | 0.24     | 60     |
| **Weighted avg** | 0.31    | 0.37   | 0.33     | 60     |

### Interprétation

- Le modèle **prédit mieux les CDI**, mais a des difficultés avec les CDD et les indépendants, probablement à cause d’un **déséquilibre des classes** et d’un dataset relativement petit.  
- L’**accuracy faible (≈37%)** indique que le modèle actuel **n’est pas très performant** pour toutes les classes.  
- La **confusion matrix** montre que la majorité des erreurs concernent les CDD et les indépendants mal classés comme CDI.  
- Le **classification report** confirme que la précision, le recall et le f1-score sont très faibles pour CDD et Indépendant, et plus élevés pour CDI.

---

## 6️⃣ Prédiction pour de nouvelles données

Exemple de prédiction sur trois nouveaux employés :

| Individu | Age | Revenu | Ancienneté | Dépense | Secteur   | Région       | Contrat prédit |
|-----------|-----|--------|------------|---------|-----------|-------------|----------------|
| 1         | 30  | 2000   | 2          | 12000   | Education | Dakar       | CDI            |
| 2         | 45  | 5000   | 10         | 25000   | IT        | Ziguinchor  | Indépendant    |
| 3         | 28  | 1800   | 1          | 8000    | Commerce  | Dakar       | CDI            |

---

## 7️⃣ Conclusion et pistes pour la suite

- Le notebook illustre une **classification supervisée complète** : exploration, prétraitement, pipeline et évaluation.  
- Les résultats montrent que le modèle **Random Forest peut être limité sur ce dataset**, surtout pour les classes minoritaires.  

### Pistes d’amélioration

1. **Augmenter la taille du dataset** ou appliquer des techniques de **rééchantillonnage** (SMOTE, oversampling, undersampling).  
2. Tester **d’autres modèles** : Logistic Regression, KNN, Gradient Boosting, XGBoost…  
3. Optimiser les **hyperparamètres** via GridSearchCV ou RandomizedSearchCV.  
4. Explorer des **features supplémentaires** ou des transformations (ex : interactions, polynômes).  
5. Évaluer le modèle avec des métriques adaptées aux **classes déséquilibrées** : macro F1-score, balanced accuracy.

---

## 8️⃣ Conclusion finale

Ce projet fournit une base solide pour expérimenter la classification de contrats d’employés, tester différents modèles, et analyser les performances par classe, tout en gardant un pipeline **réutilisable et facilement adaptable** à d’autres datasets.


  

