#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import json
import sklearn as skl
import matplotlib.pyplot as mtp
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[3]:


aria = pd.read_csv('/Users/PC2/Desktop/cours/projet perso/prediction_cause_racines/accidents-tous-req10905-2-.csv',header=5, index_col=[0], quotechar='"', sep=';', encoding='latin1')


# In[3]:


print(type(aria))


# # PRÉPARATION ET NETTOYAGE DES DONNÉES 

# In[4]:


aria.info()


# In[5]:


(aria.isnull().sum())


# In[4]:


#remplissage des NA 
aria.fillna('non communiqué', inplace = True)


# In[94]:


aria.head(20)


# ## pretraitement du texte

# In[5]:


import re
from nltk.corpus import stopwords


# On va se concentrer sur les colonnes pertinentes pour la suite de cette analyse 

# In[95]:


#sélection des colonnes pertinentes 

data_interet = aria[["Type d'accident", "Type évènement", "Matières", 
                    "Equipements", "Classe de danger CLP" , "Root causes", 
                    "Disturbances", "Conséquences"]].copy()#copie du dataframe original pour éviter toute confusion


# In[10]:


data_interet.info()


# In[96]:


""""
créer une fonction qui permet de nettoyer les colonnes textuels 
""""
def nettoyage(dataframe):
    for element in dataframe:
        # Vérifie si la colonne est de type texte avant d'appliquer des transformations
        if dataframe[element].dtype == "object":
            dataframe[element] = dataframe[element].str.lower()  # Convertit en minuscules
            dataframe[element] = dataframe[element].str.strip()  # Supprime les espaces en début et fin
            #dataframe[element] = dataframe[element].str.replace(r'[^\w\s]', '', regex=True)  # Supprime la ponctuation
    print('done !')

nettoyage(data_interet)


# la majorité des colonnes possèdent plusieurs étiquettes par exemple : 
#     Exemple : "Explosion, Incendie" ou "Gaz toxique, Liquide inflammable".
# 
# Ces valeurs ne sont pas directement exploitables pour des analyses statistiques, des corrélations, ou des modèles prédictifs.
# 
# la Solution Apportée ici est la Transformation MultiLabel
# 
# MultiLabelBinarizer transforme chaque étiquette présente dans une colonne en une variable binaire (0 ou 1).
# Chaque nouvelle colonne représente une étiquette unique et indique si elle est présente (1) ou absente (0) pour une ligne donnée.

# In[97]:


from sklearn.preprocessing import MultiLabelBinarizer

def binazer(dataframe, colonnes_interet):
    """
    Applique MultiLabelBinarizer sur les colonnes spécifiées dans colonnes_interet.

    Args:
    - dataframe (pd.DataFrame): DataFrame d'entrée.
    - colonnes_interet (list): Liste des colonnes à transformer.

    Returns:
    - pd.DataFrame: DataFrame d'origine avec les colonnes binarisées ajoutées.
    """
    mlb = MultiLabelBinarizer()
    encoded_frames = []  # Liste pour stocker les résultats encodés

    for col in colonnes_interet:
        if col in dataframe.columns:
            print(f"Traitement de la colonne : {col}")

            # Transformation des valeurs en listes
            dataframe[col] = dataframe[col].astype(str).str.lower().str.split(',').apply(
                lambda x: [i.strip() for i in x] if isinstance(x, list) else []
            )

            # Appliquer le MultiLabelBinarizer
            etiquettes_encodees = pd.DataFrame(
                mlb.fit_transform(dataframe[col]),
                columns=[f"{col}_{cls}" for cls in mlb.classes_],
                index=dataframe.index
            )
            encoded_frames.append(etiquettes_encodees)

    # Fusionner toutes les colonnes encodées
    if encoded_frames:
        encoded_df = pd.concat(encoded_frames, axis=1)
        dataframe = pd.concat([dataframe, encoded_df], axis=1)

    return dataframe


# ### 📈 Intérêts Clés de la Fonction binazer

# #### Structuration des Données Catégorielles Complexes :

# Transforme des données multi-étiquettes en un format tabulaire clair et interprétable.

# #### Compatibilité avec les Modèles Statistiques et Machine Learning :

# Les algorithmes statistiques et les modèles de Machine Learning exigent souvent des variables numériques.
# Le format binaire permet d'intégrer directement ces variables dans des modèles.

# #### Amélioration de l’Analyse Exploratoire (EDA) :

# Facilite les graphiques et les comparaisons : tu peux désormais compter et comparer la fréquence de chaque étiquette.
# Permet d'analyser la corrélation entre différentes étiquettes.

# #### Création de Nouvelles Variables :

# Chaque étiquette devient une nouvelle variable indépendante.

# Cela permet une meilleure granularité dans l’analyse.

# #### Identification des Étiquettes Dominantes :

# Permet de déterminer quelles étiquettes sont les plus fréquentes et comment elles interagissent avec d'autres variables.

# In[76]:


colonnes_interet = ["Type d'accident", "Type évènement", "Matières", 
                    "Equipements", "Classe de danger CLP", "Root causes", 
                    "Disturbances", "Conséquences"]

data_transformed = binazer(data_interet, colonnes_interet)


# In[45]:


data_transformed.info()


# In[98]:


data_transformed.head(5)


# In[16]:


print(data_transformed.dtypes)


# In[99]:


# Identifier les colonnes contenant des listes ou des types non hashables
non_hashable_cols = [
    col for col in data_transformed.columns
    if data_transformed[col].apply(lambda x: isinstance(x, list)).any()
]

print(f"Colonnes contenant des listes ou des objets non hashables : {non_hashable_cols}")


# In[100]:


for col in non_hashable_cols:
    data_transformed[col] = data_transformed[col].apply(str)


# In[101]:


# Vérification des doublons

# Vérifier les doublons
duplicates = data_transformed.duplicated()
print(f"Nombre de lignes dupliquées : {duplicates.sum()}")


# In[102]:


#suppression des doublons 
data_transformed.drop_duplicates
print(f"nombre de ligne après suppression : {len(data_transformed)}")


# In[103]:


print(data_transformed.info())


# In[79]:


data_num = data_transformed.select_dtypes(include=['number'])


# In[22]:


for col in data_num.columns : 
    print(col)


# In[53]:


frequence = data_num.sum(axis=0)
print(frequence.sort_values(ascending=False))


# # Analyse exploratoire des données EDA

# ## Observation des étiquettes les plus présentes dans l'analyse 

# In[80]:


def analyse_etiquette (dataframe) : 
    resultats = {}
    for col in dataframe.columns : 
        if "_" in col :
            prefixe = col.split("_")[0]
            if prefixe not in resultats : 
                resultats [prefixe] = []
            resultats[prefixe].append(col)
        
    analyse = {}
    for prefixe, colonnes in resultats.items(): 
        sommes = dataframe[colonnes].sum().sort_values(ascending=False)
        analyse[prefixe] = sommes
        
    return analyse


# In[81]:


analyse_etiket = analyse_etiquette(data_num)


# In[26]:


print(analyse_etiket)


# In[82]:


for cle, valeur in analyse_etiket.items():
    # Trier les fréquences de l'étiquette en ordre décroissant et sélectionner les 20 premières
    valeur.sort_values(ascending=False).head(20).plot(kind='bar', figsize=(10, 5))
    
    # Ajouter un titre au graphique
    mtp.title(f"Fréquence des 20 étiquettes les plus fréquentes pour '{cle}'")
    
    # Afficher le graphique
    mtp.show()


# on constate que pour les bariables equipement, root cause, disturbance et matières, beaucoup d'informations sont manquantes 

# In[105]:


#calcul du taux d'informations manquantes 
for col in data_interet.columns :
    taux_nan = (data_interet[col]=="non communiqué").mean()*100
    print(f"{col} : {taux_nan :.2f}% des valeurs sont 'non communiqué'")


# sans compter le fait que certaines informations sont reportées comme inconnue

# In[106]:


pourcentages_inconnu = {}

for col in data_interet.columns:
    # Trouver les lignes contenant 'inconnu'
    inconnu = data_interet[col].str.contains(
        r'inconnu[\w]*',  # Regex pour toutes les variantes de 'inconnu'
        case=False,       # Insensible à la casse
        regex=True,       # Utilise les regex
        na=False          # Ignore les valeurs NaN
    )
    
    # Calculer le pourcentage
    pourcentage_inconnu = inconnu.mean() * 100
    
    # Ajouter le résultat au dictionnaire
    pourcentages_inconnu[col] = pourcentage_inconnu

# Afficher les résultats
for col, pourcentage in pourcentages_inconnu.items():
    print(f"{col} : {pourcentage:.2f}% des valeurs contiennent 'inconnu'")


# En plus nous avons 12% des matières indiquées comme inconnues 

# ### comparaison avant et après exclusion des NAN 

# In[113]:


variables = ["Equipements", "Root causes", "Matières"]
# Filtrer les lignes où la modalité n'est pas "Non Communiquée"
data_interet_clean = data_interet.copy()
for col in variables:
    data_interet_clean = data_interet_clean[
        data_interet_clean[col] != "non communiqué"
    ]
    
# 🛠️ Étape 2 : Distribution des modalités avant exclusion

top_n = 50  # Nombre de modalités à afficher

for col in variables:
    mtp.figure(figsize=(10, 4))
    data_interet[col].value_counts().head(top_n).plot(
        kind='bar', 
        title=f'Top {top_n} des modalités - {col}'
    )
    mtp.xlabel(col)
    mtp.ylabel('Nombre d\'occurrences')
    mtp.show()
# 🛠️ Étape 3 : Distribution des modalités après exclusion
for col in variables:
    mtp.figure(figsize=(10, 4))
    data_interet_clean[col].value_counts().head(top_n).plot(
        kind='bar', 
        title=f'Top {top_n} des modalités - {col}'
    )
    mtp.xlabel(col)
    mtp.ylabel('Nombre d\'occurrences')
    mtp.show()


# après exclusion des valeurs inconnues, on constate que les grandes tendances des variables problématiques sont conservées

# ***pour la suite de cette analyse nous allons nous concentrer sur le data sans valeur inconnue pour garantir une analyse plus précise et fiable. Les valeurs inconnues, lorsqu'elles dominent une variable, peuvent biaiser les distributions, fausser les tests statistiques et perturber les axes factoriels dans une ACM. Cette exclusion permet de concentrer l'analyse sur les relations significatives entre les modalités renseignées. Néanmoins, une comparaison avant/après exclusion a été réalisée pour s'assurer de la robustesse des résultats.*** 
# 
# ***par la suite dans une seconde partie, nous réaliserons une autre analyse afin d'identifier les causes racines absentes*** 

# In[114]:


colonnes_interet = ["Type d'accident", "Type évènement", "Matières", 
                    "Equipements", "Classe de danger CLP", "Root causes", 
                    "Disturbances", "Conséquences"]

data_transformed = binazer(data_interet_clean, colonnes_interet)

data_num = data_transformed.select_dtypes(include=['number'])


# # analyse bivariée

# In[110]:


#transformer le dico en dataframe 
df_analyse_etiket = pd.DataFrame(analyse_etiket)


# In[111]:


df_analyse_etiket.describe(include="all")


# In[29]:


df_analyse_etiket.info()


# In[115]:


colonnes_avec_prefixe = [col for col in data_num if "_" in col]
print(colonnes_avec_prefixe)


# In[116]:


data_filtre = data_num[colonnes_avec_prefixe]
data_filtre.head(5)


# In[117]:


# 1. Identifier les préfixes des colonnes
prefixes = set(col.split('_')[0] for col in data_filtre.columns if '_' in col)

# 2. Initialiser un DataFrame vide pour stocker les résultats
data_filtre_ts = pd.DataFrame(index=data_filtre.index)

# 3. Boucler sur chaque préfixe et transformer
for prefix in prefixes:
    # Filtrer les colonnes correspondant au préfixe
    columns_with_prefix = [col for col in data_filtre.columns if col.startswith(prefix + '_')]

    # Identifier les modalités associées (colonne avec la valeur 1)
    modalites = data_filtre[columns_with_prefix].idxmax(axis=1).str.split('_').str[1]

    # Ajouter au DataFrame transformé
    data_filtre_ts[prefix] = modalites


# In[118]:


data_filtre_ts.head(50)


# In[175]:


data_filtre_ts.info()

# Créer des graphiques bivariés pour chaque paire de variables qualitatives
def top_modalities(data, col, top_n):
    """
    Filtre les données pour ne conserver que les top_n modalités les plus fréquentes d'une colonne.
    """
    top_values = data[col].value_counts().nlargest(top_n).index
    
    return data[data[col].isin(top_values)]for col1 in data_filtre_ts:
    top_n=20
    for col2 in data_filtre_ts:
        if col1 != col2:
            # Filtrer pour les top_n modalités des deux colonnes
            filtered_data = top_modalities(data_filtre_ts, col1, top_n=20)
            filtered_data = top_modalities(data_filtre_ts, col2, top_n=20)
                
            # Calculer les proportions pour le graphique bivarié
            proportions = filtered_data.groupby([col1, col2]).size().nlargest(top_n).reset_index(name='Count')
            proportions['Proportion'] = proportions['Count'] / proportions.groupby(col1)['Count'].transform('sum') * 100

            # Créer un graphique en barres empilées pour la paire de variables
            mtp.figure(figsize=(10, 5))
            sns.barplot(x=col1, y='Proportion', hue=col2, data=proportions)
            mtp.title(f"Graphique bivarié entre {col1} et {col2} (proportions)")
            mtp.ylabel('Proportion (%)')
            mtp.xticks(rotation=30)
            mtp.show()
# ## confirmer les observations avec un test du ki 2 

# In[119]:


data_filtre_ts = data_filtre_ts.reset_index(drop=True)
data_filtre_ts.head(5)
#data_filtre_ts_sanstitre = data_filtre_ts.drop(columns=['Titre'])


# In[120]:


#convertir les colonnes en catégorique pour optimiser le temps de calcul 
for col in data_filtre_ts.columns:
    data_filtre_ts[col] = data_filtre_ts[col].astype('category')


# In[121]:


#vérifier les modalités des mes colonnes 
for col in data_filtre_ts.columns : 
    print(f'nombre de modalité pour {col} :{data_filtre_ts[col].nunique()}')


# In[183]:


#regrouper les modalités les plus rares pour équipements et matière 
seuil = 200
for col in ['Equipements','Matières'] : 
    counts = data_filtre_ts[col].value_counts()
    rares = counts[counts<seuil].index
    data_filtre_ts[col] = data_filtre_ts[col].apply(lambda x: 'Autres' if x in rares else x)


# In[184]:


#revérifier les modalités des mes colonnes 
for col in data_filtre_ts.columns : 
    print(f'nombre de modalité pour {col} :{data_filtre_ts[col].nunique()}')


# In[185]:


# Reconvertir après regroupement
for col in data_filtre_ts.columns:
    data_filtre_ts[col] = data_filtre_ts[col].astype('category')


# In[124]:


from scipy.stats import chi2_contingency

# Initialiser les DataFrame pour les coefficients de Cramér et les p-values
cramer_v_df = pd.DataFrame(index=data_filtre_ts.columns, columns=data_filtre_ts.columns)
p_value_df = pd.DataFrame(index=data_filtre_ts.columns, columns=data_filtre_ts.columns)
#tschuprow_t_df = pd.DataFrame(index=data_filtre_ts, columns=data_filtre_ts)


# In[125]:


p_value_df.head(5)


# In[126]:



# Calculer le test de chi-deux pour chaque paire de variables qualitatives
for i, column1 in enumerate(data_filtre_ts):
    for j, column2 in enumerate(data_filtre_ts):
        if column1 != column2:
            contingency_table = pd.crosstab(data_filtre_ts[column1], data_filtre_ts[column2])
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            cramer_v = np.sqrt(chi2 / (data_filtre_ts.shape[0] * (min(contingency_table.shape) - 1)))
            #tschuprow_t = cramer_v * np.sqrt((contingency_table.shape[0] - 1) * (contingency_table.shape[1] - 1) / (data_filtre_ts.shape[0] - 1))
            cramer_v_df.loc[column1, column2] = cramer_v
            #tschuprow_t_df.loc[column1, column2] = tschuprow_t
            p_value_df.loc[column1, column2] = p

            
 
# Afficher la DataFrame des p-values
print("\nDataFrame des p-values :")
p_value_df.style.set_properties(**{'border-color': 'black', 'border-width': '1px', 'border-style': 'solid'}) 


# In[127]:


# Afficher la DataFrame des coefficients de Cramér
print("DataFrame des coefficients de Cramér :")
cramer_v_df.style.background_gradient(cmap='Greens', high=0.4, low=0).set_properties(**{'border-color': 'black','border-width': '1px', 'border-style': 'solid'})


# V de cramer montre des résultats relativement faibles, on constate qu'il y a peu d'association entre les variables. 
# ***La force d'association la plus élevée est entre le type d'accident et le type d'évènement 
# concernant les root causes (ce qui nous interresse ici), on observe pas de force d'association élevée.*** 

# # ACM 

# In[2]:


#importer le package fananalisys
get_ipython().system(' pip install prince')


# In[2]:


import prince


# In[128]:


import fanalysis.mca as mca


# In[5]:


get_ipython().system('pwd')


# In[160]:


X = data_filtre_ts.values


# In[186]:


df=data_filtre_ts
# Créer une instance de la classe MCA
my_mca = mca.MCA(row_labels=df.index.values, var_labels=df.columns)


# In[187]:


my_mca.fit(X)


# In[188]:


print(my_mca.eig_)


# In[189]:


# Graphique des valeurs propres 

my_mca.plot_eigenvalues()


# In[190]:


# Pourcentage de variance expliqué
my_mca.plot_eigenvalues(type="percentage")


# In[191]:


# Variance expliquée cumulée
my_mca.plot_eigenvalues(type="cumulative")


# 70% du pourcentage cumulé de variance est expliqué par environ 230 axes
# 
# Il y a une grande diversité dans les données, avec beaucoup de variabilité répartie sur de nombreux axes.
# Les 230 premiers axes principaux contiennent 70% de l'information totale.
# Cela veut dire que :
# L'information est très dispersée sur un grand nombre d'axes.
# Aucune petite poignée d'axes (par exemple, les 5 ou 10 premiers) ne suffit à capturer l'essentiel de l'information.
# Les 230 axes sont nécessaires pour garder une vision globale fiable des relations entre variables et individus.
# 
# ***nous allons travailler sur les 180 premiers axes qui expliqueraient 60% d'inertie tout en évitant de trop complexifier l'interprétations***

# In[193]:


df=data_filtre_ts
# Créer une instance de la classe MCA
my_mca = mca.MCA(row_labels=df.index.values, var_labels=df.columns, n_components=180)
my_mca.fit(X)


# In[34]:


df_rows = my_mca.row_topandas()
print(df_rows)


# In[35]:


print("Premier Axe")

my_mca.plot_row_cos2(num_axis=1)
print("Deuxième  Axe")

my_mca.plot_row_cos2(num_axis=2)


# In[194]:


print("Premier Axe")

my_mca.plot_col_cos2(num_axis=1)
print("Deuxième  Axe")

my_mca.plot_col_cos2(num_axis=2)
print("Troisième  Axe")

my_mca.plot_col_cos2(num_axis=3)


# In[140]:


print(dir(my_mca))


# In[196]:


my_mca.plot_col_contrib(num_axis=1)
##nb_values=20


# In[173]:


my_mca.plot_col_contrib(num_axis=2, nb_values=20)


# In[174]:


my_mca.plot_col_contrib(num_axis=3, nb_values=20)


# In[197]:


my_mca.mapping_col(num_x_axis=1, num_y_axis=2, figsize=(16, 12))


# Variabilité expliquée très faible :
# Les deux premiers axes ne parviennent pas à capter une proportion significative de la variance totale.
# Cela signifie qu'il n'y a pas de structure dominante ou de facteur clé qui organise les modalités de manière claire.
# 
# Répartition diffuse des associations :
# Les modalités ne semblent pas se regrouper naturellement autour de quelques axes principaux.
# Il est probable que les relations pertinentes soient diluées sur un nombre élevé d'axes.
# 
# Pistes d'amélioration :
# Explorer les axes supplémentaires : Les axes suivants pourraient contenir des informations plus intéressantes.
# Réduire le nombre de modalités : Fusionner ou regrouper les modalités rares sous une catégorie "Autres".
# Réévaluer les variables incluses : Certaines variables pourraient ajouter du bruit inutile à l'analyse.

# In[198]:


my_mca.mapping_col(num_x_axis=3, num_y_axis=4, figsize=(16, 12))


# In[199]:


my_mca.mapping_col(num_x_axis=5, num_y_axis=6, figsize=(16, 12))


# # prédiction des causes racines 

# In[5]:


df_na = aria[aria["Root causes"] == 'non communiqué'][["Contenu", "Root causes"]].copy()


# In[6]:


df_na.head(10)


# In[7]:


#import de spacy 
import spacy


# In[10]:


#import du paramétrage francais 
nlp = spacy.load("fr_core_news_md")


# In[9]:


"""
créer une fonction qui permet de nettoyer les colonnes textuels en utilisant spacy
"""

def nettoyer_txt_spacy (text) : 
    if pd.isna(text):
        return ''  # Gérer les valeurs NaN en les remplaçant par une chaîne vide
    
    # Étape 1 : Prétraitement de base
    text = text.lower().strip()
   
    
    #étape 2 tokénisation 
    mots_nets = []  # Liste pour stocker les tokens filtrés et lemmatisés
    doc = nlp(text)
    for token in doc:
        if not token.is_stop and token.is_alpha:
            mots_nets.append(token.lemma_)  # Ajouter le lemme du token
    
    
    # Étape 3 : Reconstruction du texte
    return ' '.join(mots_nets)


# In[10]:


#appliquer la tokenisation au dataframe 
for col in df_na : 
    if df_na[col].dtype == "object":
        df_na[col] = df_na[col].apply(lambda x: nettoyer_txt_spacy(x))


# In[15]:


df_na.head(5)


# In[26]:


#créer un nuage de mot 

from wordcloud import WordCloud

## wordcloud ne fonctionne pas sur dataframe,
## solution : Concaténer les lignes de la colonne en une chaîne unique
text = ' '.join(df_na['Contenu'].dropna())


wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
mtp.figure(figsize=(10, 6))
mtp.imshow(wordcloud, interpolation='bilinear')
mtp.axis('off')
mtp.show()


# In[27]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words=["le","la","les","l'","un","une","des","dun"])
X = vectorizer.fit_transform(df_na['Contenu'].dropna())

bigrams = pd.DataFrame(X.sum(axis=0), columns=vectorizer.get_feature_names_out()).T.sort_values(0, ascending=False).head(10)
print(bigrams)


# In[28]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([text])
print(X.shape)


# In[29]:


#observation des termes avec le score TF le plus élevé 
# Récupérer les termes et les scores TF-IDF
terms = vectorizer.get_feature_names_out()
score = X.toarray().flatten() #transforme la matrice tf-idf en un tableau numérique

# Créer un DataFrame pour visualiser les scores
df_tfidf = pd.DataFrame({'terme' : terms, 'score':score}).sort_values(by='score', ascending=False).head(10)

# Afficher les 10 termes les plus importants
print(df_tfidf)


                  


# # modèles prédictifs 

# ***ce qu'on cherche à prédire est les root cause donc la variable à prédire Y est la variable root cause***

# In[79]:


#on code la variable à prédire par une variable binaire 
df_root_cause = aria[["Contenu", "Root causes"]].copy()


# In[6]:


"""
créer une fonction qui permet de nettoyer les colonnes textuels en utilisant spacy
"""

def nettoyer_txt_spacy (text) : 
    if pd.isna(text):
        return ''  # Gérer les valeurs NaN en les remplaçant par une chaîne vide
    
    # Étape 1 : Prétraitement de base
    text = text.lower().strip()
   
    
    #étape 2 tokénisation 
    mots_nets = []  # Liste pour stocker les tokens filtrés et lemmatisés
    doc = nlp(text)
    for token in doc:
        if not token.is_stop and token.is_alpha:
            mots_nets.append(token.lemma_)  # Ajouter le lemme du token
    
    
    # Étape 3 : Reconstruction du texte
    return ' '.join(mots_nets)


# In[80]:


for col in df_root_cause : 
    if df_root_cause[col].dtype == "object":
        df_root_cause[col] = df_root_cause[col].apply(lambda x: nettoyer_txt_spacy(x))


# Ici ne pas oublier que nous avons énormément de root causes non communiquées, ce qui va fausser les modèles d'apprentissages, il est judicieux de séparer en deux df les communiqués et non communiqués. les communiqués serviront de train et test

# In[56]:


# Séparer les données étiquetées et non étiquetées
df_rc = df_root_cause[df_root_cause['Root causes'] != 'non communiquer'].copy()
df_rc_na = df_root_cause[df_root_cause['Root causes'] == 'non communiquer'].copy()

# Afficher les tailles des jeux de données
print(f"Nombre de lignes pour l'entraînement : {len(df_rc)}")
print(f"Nombre de lignes pour la prédiction : {len(df_rc_na)}")


# In[57]:


#identifier les root causes majoritaire 
count_rc = df_rc['Root causes'].value_counts().head(30)
print(count_rc)


# on constate que une classe est surreprésentée et qu'elle écrase toutes les autres 
# Il est judicieux d'appliquer un sous echantillonnage et de regrouper les classes trés minoritaires 

# In[58]:


# Définir un seuil de regroupement des classes minoritaires
seuil = 20

# Compter les occurrences des classes dans 'Root causes'
count_rc = df_rc['Root causes'].value_counts()

# Identifier les classes à regrouper (celles ayant moins de `seuil` occurrences)
classes_a_regrouper = count_rc[count_rc <= seuil].index

# Remplacer ces classes par "Autres"
df_rc['Root causes regroupées'] = df_rc['Root causes'].apply(
    lambda x: x if x not in classes_a_regrouper else 'Autres'
)

# Vérifier la nouvelle répartition
print(df_rc['Root causes regroupées'].value_counts())


# ### méthode par sous-échantillonnage 

# In[59]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter

#définir X et Y 
X = df_rc['Contenu']
Y = df_rc['Root causes regroupées']

"""
Transformation des Root causes en valeurs numériques :
pour que les modèles supervisés puissent traiter les étiquettes textuelles.
"""
#encodage de la variable cible 
encode_y = LabelEncoder()
y_encoded = encode_y.fit_transform(Y) 

class_counts = Counter(y_encoded)

# Définir un nombre minimum pour chaque classe
sampling_strategy = {cls: min(count, 500) for cls, count in class_counts.items()}

# Appliquer le sous-échantillonnage AVANT le split
rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
X_resampled, y_resampled = rus.fit_resample(X.to_numpy().reshape(-1, 1), y_encoded)

# Convertir `X_resampled` en série Pandas pour garder une structure cohérente
X_resampled = pd.Series(X_resampled.flatten())


# In[60]:


#observation avant sous echnatillonnage 
mtp.figure(figsize=(12, 6))
df_rc['Root causes regroupées'].value_counts().plot(kind='bar', color='skyblue')
mtp.title("📊 Répartition des Classes AVANT Sous-Échantillonnage")
mtp.xlabel("Classe")
mtp.ylabel("Nombre d'occurrences")
mtp.xticks(rotation=90)
mtp.show()

#observation après sous échantillonnage 
mtp.figure(figsize=(12, 6))
pd.Series(y_resampled).value_counts().plot(kind='bar', color='orange')
mtp.title("📊 Répartition des Classes APRÈS Sous-Échantillonnage")
mtp.xlabel("Classe")
mtp.ylabel("Nombre d'occurrences")
mtp.xticks(rotation=90)
mtp.show()


# In[61]:


#séparation 80/20
# Séparation en 80% train et 20% test
x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Vérifier la répartition des classes dans train et test
print("Répartition y_train :", Counter(y_train))
print("Répartition y_test :", Counter(y_test))


# In[62]:


#séparation apprentissage/validation 
from sklearn.feature_extraction.text import TfidfVectorizer


trans_vect = TfidfVectorizer()
"""
Le texte brut est vectorisé en utilisant TF-IDF, 
ce qui est indispensable pour que les modèles puissent interpréter les données textuelles.
"""

x_train_trans = trans_vect.fit_transform(x_train)
x_test_trans = trans_vect.transform(x_test)


# In[66]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
#on définit deux modèeles 
modele_bayes = MultinomialNB(force_alpha=True)  # Evite les erreurs dues aux classes rares
modele_svm = SVC()


# In[67]:


from sklearn.metrics import accuracy_score, classification_report

modele_bayes.fit(x_train_trans,y_train)

# Prédictions
y_pred = modele_bayes.predict(x_test_trans)


# In[68]:


# Évaluer le modèle
print("Précision :", accuracy_score(y_test, y_pred))


# In[35]:


print(classification_report(y_test, y_pred))


# Les colonnes du rapport contiennent les métriques suivantes pour chaque classe :
# 
# Precision : Proportion des prédictions correctes pour une classe donnée parmi toutes les prédictions pour cette classe.
# Recall (Rappel) : Proportion des instances correctement prédites pour une classe donnée parmi toutes les instances réelles de cette classe.
# F1-score : Moyenne harmonique de la précision et du rappel (indique l'équilibre entre les deux).

# ici on voit clairement que La classe 899 domine le jeu de données (3526 exemples sur 5449).
# Le modèle est très performant pour cette classe avec un recall (1.00), mais cela peut indiquer un biais en faveur des classes majoritaires.

# Déséquilibre des Classes :
# Une seule classe (899) domine le jeu de données, ce qui explique pourquoi le modèle performe bien pour cette classe mais ignore presque toutes les autres.
# Précision biaisée :
# La précision globale (64,7 %) est biaisée par la classe majoritaire et ne reflète pas la capacité du modèle à gérer les classes minoritaires.
# Mauvaises performances sur les classes minoritaires :
# Pour la plupart des classes, precision, recall et F1-score sont à 0.00, ce qui signifie que le modèle ne fait pas mieux qu’un choix aléatoire pour ces classes.

# In[69]:


#relance avec un nouveau seuil
# Définir un seuil de regroupement des classes minoritaires
seuil = 100

# Compter les occurrences des classes dans 'Root causes'
count_rc = df_rc['Root causes'].value_counts()

# Identifier les classes à regrouper (celles ayant moins de `seuil` occurrences)
classes_a_regrouper = count_rc[count_rc <= seuil].index

# Remplacer ces classes par "Autres"
df_rc['Root causes regroupées'] = df_rc['Root causes'].apply(
    lambda x: x if x not in classes_a_regrouper else 'Autres'
)

# Vérifier la nouvelle répartition
print(df_rc['Root causes regroupées'].value_counts())

"""
**********************
"""

#définir X et Y 
X = df_rc['Contenu']
Y = df_rc['Root causes regroupées']

"""
Transformation des Root causes en valeurs numériques :
pour que les modèles supervisés puissent traiter les étiquettes textuelles.
"""
#encodage de la variable cible 
encode_y = LabelEncoder()
y_encoded = encode_y.fit_transform(Y) 

class_counts = Counter(y_encoded)

# Définir un nombre minimum pour chaque classe
sampling_strategy = {cls: min(count, 500) for cls, count in class_counts.items()}

# Appliquer le sous-échantillonnage AVANT le split
rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
X_resampled, y_resampled = rus.fit_resample(X.to_numpy().reshape(-1, 1), y_encoded)

# Convertir `X_resampled` en série Pandas pour garder une structure cohérente
X_resampled = pd.Series(X_resampled.flatten())


# In[70]:


#observation avant sous echnatillonnage 
mtp.figure(figsize=(12, 6))
df_rc['Root causes regroupées'].value_counts().plot(kind='bar', color='skyblue')
mtp.title("📊 Répartition des Classes AVANT Sous-Échantillonnage")
mtp.xlabel("Classe")
mtp.ylabel("Nombre d'occurrences")
mtp.xticks(rotation=90)
mtp.show()

#observation après sous échantillonnage 
mtp.figure(figsize=(12, 6))
pd.Series(y_resampled).value_counts().plot(kind='bar', color='orange')
mtp.title("📊 Répartition des Classes APRÈS Sous-Échantillonnage")
mtp.xlabel("Classe")
mtp.ylabel("Nombre d'occurrences")
mtp.xticks(rotation=90)
mtp.show()


# In[72]:


#séparation 80/20
# Séparation en 80% train et 20% test
x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Vérifier la répartition des classes dans train et test
print("Répartition y_train :", Counter(y_train))
print("Répartition y_test :", Counter(y_test))


# In[73]:


#séparation apprentissage/validation 
from sklearn.feature_extraction.text import TfidfVectorizer


trans_vect = TfidfVectorizer()
"""
Le texte brut est vectorisé en utilisant TF-IDF, 
ce qui est indispensable pour que les modèles puissent interpréter les données textuelles.
"""

x_train_trans = trans_vect.fit_transform(x_train)
x_test_trans = trans_vect.transform(x_test)

#relance du modèle bayes
modele_bayes.fit(x_train_trans,y_train)

y_pred = modele_bayes.predict(x_test_trans)

# Évaluer le modèle
print("Précision :", accuracy_score(y_test, y_pred))


# on va tester avec ***complementNB*** 
# mais la différence entre les classes sont trop extrême 
# on va donc continuer à sous échantillonner mais uniquement la classe majoritaire 

# In[76]:


#sous échantillonnage sur la classe majoritaire 
from imblearn.under_sampling import RandomUnderSampler

# Trouver la classe majoritaire
class_counts = Counter(y_encoded)
major_class = max(class_counts, key=class_counts.get)
under_sampler = RandomUnderSampler(sampling_strategy={major_class: 5000}, random_state=42)

# Appliquer le sous-échantillonnage
X_resampled, y_resampled = under_sampler.fit_resample(X.to_numpy().reshape(-1, 1), y_encoded)

print("Nouvelle distribution des classes :", Counter(y_resampled))


# In[77]:


from sklearn.naive_bayes import ComplementNB


# Initialisation du modèle
complement_nb = ComplementNB()

# Entraînement sur les données
complement_nb.fit(x_train_trans, y_train)

# Prédictions
y_pred = complement_nb.predict(x_test_trans)

# Évaluation du modèle
print("Accuracy :", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[83]:


#relance avec un nouveau seuil
# Définir un seuil de regroupement des classes minoritaires
seuil = 100

# Compter les occurrences des classes dans 'Root causes'
count_rc = df_rc['Root causes'].value_counts()

# Identifier les classes à regrouper (celles ayant moins de `seuil` occurrences)
classes_a_regrouper = count_rc[count_rc <= seuil].index

# Remplacer ces classes par "Autres"
df_rc['Root causes regroupées'] = df_rc['Root causes'].apply(
    lambda x: x if x not in classes_a_regrouper else 'Autres'
)

# Vérifier la nouvelle répartition
print(df_rc['Root causes regroupées'].value_counts())

"""
**********************
"""

#définir X et Y 
X = df_rc['Contenu']
Y = df_rc['Root causes regroupées']

"""
Transformation des Root causes en valeurs numériques :
pour que les modèles supervisés puissent traiter les étiquettes textuelles.
"""
#encodage de la variable cible 
encode_y = LabelEncoder()
y_encoded = encode_y.fit_transform(Y) 

class_counts = Counter(y_encoded)

# Identifier la classe majoritaire et les classes minoritaires
major_class = max(class_counts, key=class_counts.get)  # Classe la plus fréquente
print(f"Classe majoritaire identifiée : {major_class}, Nombre d'occurrences : {class_counts[major_class]}")


# En sous échantillonnant la classe majoritaire on risque de perdre trop d'information, à côté de ça les autres classes sont faibles... ***Pour compenser on va sur échantillonner les autres classes***

# In[88]:


from imblearn.over_sampling import SMOTE

# ✅ Étape 1 : Vectorisation du texte
vectorizer = TfidfVectorizer(max_features=5000)  # On limite à 5000 features pour éviter la surcharge
X_vectorized = vectorizer.fit_transform(X).toarray()  # Conversion en array pour compatibilité

# ✅ Étape 2 : Sous-échantillonnage de la classe majoritaire
under_sampler = RandomUnderSampler(sampling_strategy={major_class: 5000}, random_state=42)
X_under, y_under = under_sampler.fit_resample(X_vectorized, y_encoded)

print("Répartition après sous-échantillonnage :", Counter(y_under))

# ✅ Étape 3 : Sur-échantillonnage des classes minoritaires (moins de 500 occurrences)
minor_classes = {cls: 500 for cls, count in Counter(y_under).items() if count < 500}
smote = SMOTE(sampling_strategy=minor_classes, random_state=42)

# **Important** : SMOTE ne fonctionne qu'avec des **données numériques**, donc on utilise `X_under`
X_resampled, y_resampled = smote.fit_resample(X_under, y_under)

print("Répartition après sur-échantillonnage :", Counter(y_resampled))

# ✅ Étape 4 : **Split final**
x_train_final, x_test_final, y_train_final, y_test_final = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

print("Répartition finale dans y_train :", Counter(y_train_final))
print("Répartition finale dans y_test :", Counter(y_test_final))


# In[89]:


# Initialisation du modèle
complement_nb = ComplementNB()

# Entraînement sur les données
complement_nb.fit(x_train_final, y_train_final)

# Prédictions
y_pred = complement_nb.predict(x_test_final)

# Évaluation du modèle
print("Accuracy :", accuracy_score(y_test_final, y_pred))
print(classification_report(y_test_final, y_pred))


# In[91]:


classe_predite = encode_y.inverse_transform(y_encoded)
print(classe_predite)


# In[92]:


#avec un seuil moins élevé
seuil = 50

# Compter les occurrences des classes dans 'Root causes'
count_rc = df_rc['Root causes'].value_counts()

# Identifier les classes à regrouper (celles ayant moins de `seuil` occurrences)
classes_a_regrouper = count_rc[count_rc <= seuil].index

# Remplacer ces classes par "Autres"
df_rc['Root causes regroupées'] = df_rc['Root causes'].apply(
    lambda x: x if x not in classes_a_regrouper else 'Autres'
)

# Vérifier la nouvelle répartition
print(df_rc['Root causes regroupées'].value_counts())

"""
**********************
"""

#définir X et Y 
X = df_rc['Contenu']
Y = df_rc['Root causes regroupées']

"""
Transformation des Root causes en valeurs numériques :
pour que les modèles supervisés puissent traiter les étiquettes textuelles.
"""
#encodage de la variable cible 
encode_y = LabelEncoder()
y_encoded = encode_y.fit_transform(Y) 

class_counts = Counter(y_encoded)

# Identifier la classe majoritaire et les classes minoritaires
major_class = max(class_counts, key=class_counts.get)  # Classe la plus fréquente
print(f"Classe majoritaire identifiée : {major_class}, Nombre d'occurrences : {class_counts[major_class]}")


# In[93]:


# ✅ Étape 1 : Vectorisation du texte
vectorizer = TfidfVectorizer(max_features=5000)  # On limite à 5000 features pour éviter la surcharge
X_vectorized = vectorizer.fit_transform(X).toarray()  # Conversion en array pour compatibilité

# ✅ Étape 2 : Sous-échantillonnage de la classe majoritaire
under_sampler = RandomUnderSampler(sampling_strategy={major_class: 5000}, random_state=42)
X_under, y_under = under_sampler.fit_resample(X_vectorized, y_encoded)

print("Répartition après sous-échantillonnage :", Counter(y_under))

# ✅ Étape 3 : Sur-échantillonnage des classes minoritaires (moins de 500 occurrences)
minor_classes = {cls: 500 for cls, count in Counter(y_under).items() if count < 500}
smote = SMOTE(sampling_strategy=minor_classes, random_state=42)

# **Important** : SMOTE ne fonctionne qu'avec des **données numériques**, donc on utilise `X_under`
X_resampled, y_resampled = smote.fit_resample(X_under, y_under)

print("Répartition après sur-échantillonnage :", Counter(y_resampled))

# ✅ Étape 4 : **Split final**
x_train_final, x_test_final, y_train_final, y_test_final = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

print("Répartition finale dans y_train :", Counter(y_train_final))
print("Répartition finale dans y_test :", Counter(y_test_final))


# In[94]:


# Initialisation du modèle
complement_nb = ComplementNB()

# Entraînement sur les données
complement_nb.fit(x_train_final, y_train_final)

# Prédictions
y_pred = complement_nb.predict(x_test_final)

# Évaluation du modèle
print("Accuracy :", accuracy_score(y_test_final, y_pred))
print(classification_report(y_test_final, y_pred))


# In[95]:


#avec un seuil moins élevé
seuil = 20

# Compter les occurrences des classes dans 'Root causes'
count_rc = df_rc['Root causes'].value_counts()

# Identifier les classes à regrouper (celles ayant moins de `seuil` occurrences)
classes_a_regrouper = count_rc[count_rc <= seuil].index

# Remplacer ces classes par "Autres"
df_rc['Root causes regroupées'] = df_rc['Root causes'].apply(
    lambda x: x if x not in classes_a_regrouper else 'Autres'
)

# Vérifier la nouvelle répartition
print(df_rc['Root causes regroupées'].value_counts())

"""
**********************
"""

#définir X et Y 
X = df_rc['Contenu']
Y = df_rc['Root causes regroupées']


#encodage de la variable cible 
encode_y = LabelEncoder()
y_encoded = encode_y.fit_transform(Y) 

class_counts = Counter(y_encoded)

# Identifier la classe majoritaire et les classes minoritaires
major_class = max(class_counts, key=class_counts.get)  # Classe la plus fréquente
print(f"Classe majoritaire identifiée : {major_class}, Nombre d'occurrences : {class_counts[major_class]}")


# ✅ Étape 1 : Vectorisation du texte
vectorizer = TfidfVectorizer(max_features=5000)  # On limite à 5000 features pour éviter la surcharge
X_vectorized = vectorizer.fit_transform(X).toarray()  # Conversion en array pour compatibilité

# ✅ Étape 2 : Sous-échantillonnage de la classe majoritaire
under_sampler = RandomUnderSampler(sampling_strategy={major_class: 5000}, random_state=42)
X_under, y_under = under_sampler.fit_resample(X_vectorized, y_encoded)

print("Répartition après sous-échantillonnage :", Counter(y_under))

# ✅ Étape 3 : Sur-échantillonnage des classes minoritaires (moins de 500 occurrences)
minor_classes = {cls: 500 for cls, count in Counter(y_under).items() if count < 500}
smote = SMOTE(sampling_strategy=minor_classes, random_state=42)

# **Important** : SMOTE ne fonctionne qu'avec des **données numériques**, donc on utilise `X_under`
X_resampled, y_resampled = smote.fit_resample(X_under, y_under)

print("Répartition après sur-échantillonnage :", Counter(y_resampled))

# ✅ Étape 4 : **Split final**
x_train_final, x_test_final, y_train_final, y_test_final = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

"""************lancement du modèle**********************
"""


# Initialisation du modèle
complement_nb = ComplementNB()

# Entraînement sur les données
complement_nb.fit(x_train_final, y_train_final)

# Prédictions
y_pred = complement_nb.predict(x_test_final)

# Évaluation du modèle
print("Accuracy :", accuracy_score(y_test_final, y_pred))
print(classification_report(y_test_final, y_pred))


# ### Prédiction des root causes avec ComplementNB

# In[96]:


#prédiction du modèle complemntNB 

#définir X 
X_na = df_rc_na['Contenu']

#transformer le texte avec le même vectorizer entraîné
X_na_transformed = vectorizer.transform(X_na)

# Prédire les Root Causes
y_na_pred = complement_nb.predict(X_na_transformed)

# Convertir les labels prédits en catégories textuelles
y_na_pred_text = encode_y.inverse_transform(y_na_pred)

# Ajouter les prédictions dans le dataframe
df_rc_na['Root causes prédites'] = y_na_pred_text


# In[98]:


#visualisation des root causes
root_causes_counts = df_rc_na['Root causes prédites'].value_counts()

mtp.figure(figsize=(12, 6))

mtp.bar(root_causes_counts.index, root_causes_counts.values, color='skyblue')
mtp.xlabel("Root Causes Prédites")
mtp.ylabel("Nombre d'occurrences")
mtp.title("Distribution des Root Causes Prédites")
mtp.xticks(rotation=90)  # Faire pivoter les étiquettes si elles sont longues

mtp.show()


# In[99]:


#générer un nuage de point sur le dataframe des root causes non identifiées 
text = ' '.join(df_rc_na['Contenu'].dropna())


wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
mtp.figure(figsize=(10, 6))
mtp.imshow(wordcloud, interpolation='bilinear')
mtp.axis('off')
mtp.show()


# In[ ]:




