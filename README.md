# SiameseNetwork
Projet système de 6 mois à l'ENSTA Bretagne, filière Sytèmes d'Observation et Intelligence Artificielle, UE 6.1.

## Configuration
- Tensorflow 2.14.0

## Dossiers des splits
Tous les dossiers 'Split' contiennent les données sous format csv de train (80%), test (10%) et validation (10%). Voici les informations contenues dans les différents dossiers :
- Split_Cargo_Bulk_Container : fichiers csv avec tous les chemins vers les images Cargo, Bulk Carrier et Container Ship.

- Split_Cargo_Bulk_Container_frugal_vh : fichiers csv avec tous les chemins vers les images Cargo, Bulk Carrier et Container Ship, avec des donées frugales et la polarisation vh seulement.

- Split_Cargo_Bulk_Container_frugal_vv : fichiers csv avec tous les chemins vers les images Cargo, Bulk Carrier et Container Ship, avec des donées frugales et la polarisation vv seulement.

- Split_Tanker_Bulk_Container_frugal_vv : fichiers csv avec tous les chemins vers les images Tanker, Bulk Carrier et Container Ship, avec des donées frugales et la polarisation vv seulement.

- Split_siamese : fichiers csv pour l'entraînement du modèle siamois, ie avec des paires d'images construites à partir des données de Split_Tanker_Bulk_Container_frugal_vv.

- Split_quater_siamese : fichiers csv avec seulement un quart des données de Split_siamese.

## Fichiers Python .py pour les classes

- `Siamese_model.py` : contient la classe SiameseTrainer, qui est le modèle siamois.
- `CNN_model.py` : contient la classe ModelTrainer, qui est le modèle CNN.

## Fichiers Jupyter Notebook
- `Etude_bdd.ipynb` : étude de la base de données. On y étudie le nombre d'images par classe, la distribution des classes, la distribution des images par polarisation, etc.
- `Extraction_donnees_xml.ipynb` : extraction des données des fichiers xml Ship.xml, pour chaque catégorie de bateau. Le but est de comparer l'étiquette des images avec les données des fichiers xml, notamment la balise 'Elaborated_name'. Les images sont ensuite catégorisées dans la bonne classe.
- `Split_data.ipynb` : split des données en train, test et validation. On crée des fichiers csv avec les chemins vers les images pour chaque split.
- `Split_in_pairs.ipynb` : split des données en paires pour l'entraînement du modèle siamois. On crée des fichiers csv avec les chemins vers les paires d'images.
- `CNN_training.ipynb` : entraînement du modèle CNN.
- `Siamese_training.ipynb` : entraînement du modèle siamois.

Les autres fichiers .ipynb sont d'autres entrainements de différents réseaux SIAMOIS, avec différents paramètres. Le principe de fonctionnement est le même que pour `Siamese_training.ipynb`, et les résultats sont comparés.

## Prérequis
Pour lancer n'importe quel fichier Notebook, il faut avoir téléchargé la base de données OpenSARShip (https://opensar.sjtu.edu.cn/project.html). Les données peuvent être splitées (nouvelles catégories) avec le Notebook `Extraction_donnees_xml.ipynb`, et les données peuvent être splitées en train, test et validation avec le Notebook `Split_data.ipynb`.