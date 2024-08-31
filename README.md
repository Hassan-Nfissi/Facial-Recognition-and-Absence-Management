# Projet de Reconnaissance Faciale et Gestion des Absences

Ce projet consiste en la création d'une application web utilisant la reconnaissance faciale pour la gestion des absences des employés. L'application permet d'ajouter des employés, de les entraîner pour la reconnaissance faciale, et de surveiller les présences en temps réel.

## Langages et Technologies Utilisés

1. **HTML5/CSS3**
   - **HTML5** : Langage de base pour structurer le contenu des pages web. Chaque site web utilise HTML pour définir la structure des pages.
   - **CSS3** : Utilisé pour styliser le contenu HTML et améliorer l'apparence des pages web.

2. **JavaScript**
   - **JavaScript** : Langage de programmation orienté objet, utilisé pour rendre les pages web interactives et dynamiques. Permet de manipuler les éléments de la page, valider des formulaires, et créer des animations.

3. **Python**
   - **Python** : Langage de programmation interprété, orienté objet, connu pour sa simplicité et sa polyvalence. Utilisé ici pour le backend de l'application, notamment avec Flask.

4. **Flask**
   - **Flask** : Micro-framework en Python utilisé pour développer l'application web. Il gère les requêtes HTTP, les sessions utilisateur, et les interactions avec la base de données.

5. **Scikit-learn**
   - **Scikit-learn** : Bibliothèque Python pour l'apprentissage automatique. Utilisée pour le traitement des données et l'entraînement du modèle de reconnaissance faciale.

6. **OpenCV**
   - **OpenCV** : Bibliothèque open source pour la vision par ordinateur. Utilisée pour la détection et la reconnaissance des visages dans les images et vidéos.

7. **IP Webcam**
   - **IP Webcam** : Application Android permettant de transformer un téléphone en caméra IP. Elle est utilisée pour capturer le flux vidéo en temps réel.

8. **SQLite3**
   - **SQLite3** : Système de gestion de base de données relationnelles léger. Utilisé pour stocker les informations des employés et les données d'absence.

## Logiciels Utilisés

1. **PyCharm**
   - **PyCharm** : Environnement de développement intégré (IDE) pour Python. Utilisé pour écrire, déboguer et tester le code de l'application.

2. **UML**
   - **UML** : Langage de modélisation utilisé pour la conception visuelle du système. Permet de créer des diagrammes de structure et de comportement pour mieux comprendre et concevoir le système.

## Fonctionnalités Principales

1. **Page d'Authentification**
   - Formulaire de connexion sécurisé pour les administrateurs afin de gérer les employés et les absences.

2. **Page d'Accueil**
   - Menu de navigation permettant d'accéder aux différentes fonctionnalités comme l'ajout d'employés, l'entraînement du modèle, et la visualisation des absences.

3. **Ajouter un Employé**
   - Permet d'ajouter de nouveaux employés dans le système avec des informations telles que le prénom, le nom, le poste, et une photo. Affiche également la liste des employés enregistrés.

4. **Entraînement**
   - Capture des photos des employés pour entraîner le modèle de reconnaissance faciale. Le flux vidéo en direct permet de visualiser le processus de capture.

5. **Reconnaissance Faciale**
   - Page dédiée à la reconnaissance faciale en temps réel. Affiche un flux vidéo en direct et les informations des employés détectés.

