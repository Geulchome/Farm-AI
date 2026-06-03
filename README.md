# Application de prédiction agricole - Burundi

## Présentation

Cette application en ligne permet à un utilisateur de prédire si une parcelle au Burundi aura une bonne ou une mauvaise récolte. Elle fonctionne avec trois modèles différents :

- Arbre de décision
- Forêt aléatoire
- Régression logistique

L’interface est bâtie avec Streamlit et fournit une estimation immédiate à partir des caractéristiques de la parcelle.

## Utilisation pour l’utilisateur final

1. Ouvrez l’application dans votre navigateur.
2. Dans la barre latérale, remplissez les informations de la parcelle :
   - `Année`
   - `Saison` (A ou B)
   - `Province`
   - `Culture`
   - `Altitude (m)`
   - `Pluviométrie (mm)`
   - `Température moyenne (°C)`
   - `Superficie (ha)`
   - `Utilisation d’engrais` (0 = non, 1 = oui)
   - `Accès à l’irrigation` (0 = non, 1 = oui)
   - `Nombre de ménages`
3. Vérifiez les données saisies dans le tableau.
4. Cliquez sur le bouton `Prédire`.

## Ce que l’application affiche

L’application montre :

- la prédiction de chaque modèle : `Bonne récolte` ou `Mauvaise récolte`
- la probabilité associée à chaque prédiction

## Comment interpréter les résultats

- Si tous les modèles donnent `Bonne récolte`, la parcelle a de bonnes chances.
- Si tous les modèles donnent `Mauvaise récolte`, la parcelle est plus à risque.
- Si les résultats sont différents, il faut analyser les données avec attention.

## Lancement local

Pour lancer l’application sur votre machine :

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Déploiement en ligne (Streamlit Cloud)

1. Publiez le dossier sur GitHub.
2. Connectez-vous à Streamlit Cloud : `https://streamlit.io/cloud`
3. Créez une nouvelle app en sélectionnant :
   - le dépôt GitHub
   - la branche
   - le fichier principal `streamlit_app.py`

Après déploiement, l’application sera accessible via une URL publique.

## Structure du projet

- `streamlit_app.py` : application Web
- `requirements.txt` : dépendances Python
- `agriculture_burundi.csv` : jeu de données utilisé pour récupérer les options
- `decision_tree.joblib` : modèle Arbre de décision
- `random_forest.joblib` : modèle Forêt aléatoire
- `logistic_regression.joblib` : modèle Régression logistique
- `scaler.joblib` : transformée de normalisation
- `TP_Agriculture_Burundi.ipynb` : notebook de développement
- `TP_Agriculture_Burundi_executed.ipynb` : notebook exécuté
- `rapport_reflexion.md` : rapport de réflexion
- `.gitignore` : fichiers ignorés par Git


