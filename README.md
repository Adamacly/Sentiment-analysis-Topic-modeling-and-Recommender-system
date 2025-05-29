# Analyse d’opinion, détection de thématiques dans les avis clients et système de récommandation

Ce projet vise à analyser automatiquement les avis clients pour en extraire l’opinion (analyse de sentiment), identifier les thématiques principales (topic mining) et proposer un système de recommandation personnalisé basé sur les préférences détectées.

## 📌 Objectifs

- Classifier automatiquement les avis clients selon leur polarité (positif, négatif, neutre)
- Extraire les thématiques récurrentes à partir des avis textuels
- Représenter les résultats à l’aide de visualisations interactives
- Implémenter un système de recommandation item-based pour suggérer des produits ou services

## 🧰 Technologies utilisées

- **Python 3.10+**
- **Pandas**, **NumPy** : manipulation de données
- **NLTK**, **Pandas** : traitement du langage naturel
- **Transformers (HuggingFace)** : modèles pré-entraînés (ex : SieBERT pour l’analyse de sentiment)
- **BERTopic** : détection de thématiques
- **Item-based** : système de recommandation
- **Plotly Dash** : interface web interactive pour la visualisation et l'interaction

## 🚀 Fonctionnalités principales

1. **Analyse de sentiment**
   - Prédiction du sentiment à partir d’un avis client via SiEBERT
   - Statistiques globales sur les sentiments

2. **Détection de thématiques**
   - Extraction des sujets dominants avec BERTopic
   - Affichage dynamique des résultats (WordClouds, bar charts…)

3. **Système de recommandation**
   - Recommandation item-based
   - Entrée utilisateur pour suggestion personnalisée

4. **Application Dash**
   - Interface utilisateur simple
   - Visualisations interactives (graphes, nuages de mots, filtres...)


## 💻 Installation et utilisation

### 🔁 1. Cloner le dépôt

```bash
git clone https://github.com/Adamacly/Sentiment-analysis-Topic-modeling-and-Recommender-system.git
cd Sentiment-analysis-Topic-modeling-and-Recommender-system
```

### 🛠️ 2. Créer et activer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate  # Pour Windows : venv\Scripts\activate
```

### 📦 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 🚀 4. Lancer l'application Dash

```bash
python app.py
```
