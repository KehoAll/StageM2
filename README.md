Le dossier tensiomètre contient les codes relatifs au contrôle du dispositif expérimental.
En voici certains qui sont important :
show_measurements.py permet d'obtenir le retour en direct des capteurs du dispositif.
do_calibration.py permet de réaliser l'étalonnage des capteurs de manière complètement automatisé
Les fichiers python commençant par procedure permettent de réaliser une expérience précise comme un creep à hauteur fixé par exemple
Le fichier chirp.py est appelé lors de la phase de réponse à une rampe de fréquence.

Le dossier readlif contient le code nécessaire permettant la lecture des fichiers .lif qu'enregistre le microscope confocal permettant ainsi de traiter les vidéos sous python.

Le dossier gelrupture contient quant à lui des codes propres àn l'analyse des données:
export_ZX_YX_movie.py permet d'avoir une vidéo mp4 (ou série d'image au choix) avec vue en coupe sur les côtés du fichier .lif que sort le microscope confocal
opticalflow_GPU.py permet d'obtenir le fichier de flux optique à partir de la vidéo en .lif	
pH_evolution.py permet de simuler l'évolution du pH au cours du temps d'ne solution de caséine avec une base faible et du GDL
	
Le notebook jupyter lab dans la racine contient les différents codes d'analyse de vidéo lif allant de la recherche de maxima en passant par le calcul de la matrice hessienne, le calcul de l'épaisseur du gel, l'évolution de l'orientation de la strucure en fonction etc... donc une très grande partie des codes d'analyses
