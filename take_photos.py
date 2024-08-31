import os
import cv2
def rename_images_in_folder(folder_path):
    # Récupérer la liste des fichiers dans le dossier
    files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    # Initialiser le compteur pour les nouveaux noms de fichiers
    count = 0  # Vous pouvez ajuster le point de départ selon vos besoins

    for file_name in files:
        # Décomposer le nom de fichier et vérifier l'extension
        file_base, file_ext = os.path.splitext(file_name)

        # Vérifier si le fichier est une image (par exemple, .jpg)
        if file_ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:

            new_file_name = f"{count:05d}{file_ext}"
            old_file_path = os.path.join(folder_path, file_name)
            new_file_path = os.path.join(folder_path, new_file_name)
            os.rename(old_file_path, new_file_path)
            print(f"Renamed '{file_name}' to '{new_file_name}'")

            count += 1
def capture_photos(num_photos, output_dir='dataset/anas-yomni'):
    # Créer le répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Initialiser la capture vidéo
    cam = cv2.VideoCapture(0)  # Utilise la caméra par défaut
    if not cam.isOpened():
        print("Erreur : La caméra n'a pas pu être ouverte.")
        return

    cam.set(3, 640)  # Largeur de la vidéo
    cam.set(4, 480)  # Hauteur de la vidéo

    print(f"\n[INFO] Initialisation de la capture de {num_photos} photos. Appuyez sur 'q' pour quitter.")

    count = 0
    while count < num_photos:
        ret, frame = cam.read()
        if not ret:
            print("Erreur : Impossible de lire l'image de la caméra.")
            break

        # Nommer l'image avec des zéros à gauche pour avoir 4 chiffres
        file_name = f"{count:04d}.jpg"
        file_path = os.path.join(output_dir, file_name)

        # Enregistrer l'image
        cv2.imwrite(file_path, frame)
        print(f"Image {file_name} enregistrée.")

        # Afficher l'image capturée
        cv2.imshow('Capture', frame)

        count += 1

        # Attendre un court instant ou appuyer sur 'q' pour quitter
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"\n[INFO] Fin de la capture de photos.")
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_photos(80)


