import cv2
import numpy as np
import glob

# Definieren der Schachbrettgröße
# dh. Anzahl der Ecken im Schachbrettmuster (muss mit dem Schachbrettbild übereinstimmen)
# Größe sollte daher die Anzahl der "inneren" Ecken pro Dimension des Schachbretts sein und nicht die Anzahl der gesamten Quadrate
# bedeutet -> dass OpenCV ein Schachbrett erwartet, das 10 innere Ecken in einer Richtung (z.B. horizontal) und 8 in die  andere vertikale Richtung
schachbrett_groesse = (10, 8)

# Arrays für die Speicherung von Objektpunkten und Bildpunkten
obj_punkte = []  # 3D-Punkte in der realen Welt
bild_punkte = []  # 2D-Punkte im Bild

# Erstellen eines Gitters von Punkten
obj_punkt = np.zeros((np.prod(schachbrett_groesse), 3), np.float32)
obj_punkt[:,:2] = np.mgrid[0:schachbrett_groesse[0], 0:schachbrett_groesse[1]].T.reshape(-1, 2)

# Durchlaufen der Schachbrett-Bilder und Finden der jeweiligen Schachbrett-Ecken
bilder = glob.glob('kalibrierung/*.jpg')

for bild_pfad in bilder:
    bild = cv2.imread(bild_pfad)
    grau = cv2.cvtColor(bild, cv2.COLOR_BGR2GRAY)

    # Finden der Schachbrett-Ecken
    erfolgreich, ecken = cv2.findChessboardCorners(grau, schachbrett_groesse, None)

    if erfolgreich:
        obj_punkte.append(obj_punkt)
        bild_punkte.append(ecken)

        # Zeichnen und Anzeigen der Ecken
        cv2.drawChessboardCorners(bild, schachbrett_groesse, ecken, erfolg)
        cv2.imshow('Schachbrett-Ecken', bild)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Check, on die Bildpunkte gefunden wurden
if bild_punkte:
    # Kamerakalibrierung
    ret, kamera_matrix, verzerrungs_koeff, rotationsvektoren, translationsvektoren = cv2.calibrateCamera(obj_punkte, bild_punkte, grau.shape[::-1], None, None)

    # Anzeigen der Kalibrierungsparameter
    print("Kameramatrix:\n", kamera_matrix)
    print("Verzerrungskoeffizienten:\n", verzerrungs_koeff)

    '''
    # TODO: Rotationsvektoren und Translationsvektoren für jedes Bild anzeigen bzgl.
    # Schätzung der Position und Orientierung der Kamera relativ zum Schachbrett.
    
    print("\nRotationsvektoren und Translationsvektoren für jedes Bild:")   
    for i in range(len(rotationsvektoren)):
        print(f"\nBild {i + 1}:")
        print("Rotationsvektor:", rotationsvektoren[i])
        print("Translationsvektor:", translationsvektoren[i])

        # Rotationsvektoren in Rotationsmatrizen umwandeln, um die Orientierung der Kamera in Bezug zum Schachbrett darzustellen.
        # Umwandlung der Rotationsvektoren in Rotationsmatrizen
        R = cv2.Rodrigues(rotationsvektoren[i])[0]
        print("Rotationsmatrix:\n", R)
    '''
else:
    print("Keine Schachbrett-Ecken gefunden.")
