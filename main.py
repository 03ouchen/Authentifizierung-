import cv2

def main():
    # Öffne die Kamera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Fehler beim Öffnen der Kamera.")
        return

    # Lade den vorgefertigten Gesichts- und Augendetektor von OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    while True:
        # Lies ein Frame von der Kamera
        ret, frame = cap.read()

        if not ret:
            print("Fehler beim Lesen des Frames.")
            break

        # Konvertiere das Frame in Graustufen für die Gesichtserkennung
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Erkenne Gesichter im Bild
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Für jedes erkannte Gesicht, erkenne die Augen
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        # Zeige das Frame mit markierten Augen an
        cv2.imshow('Augenerkennung', frame)

        # Warte auf das Drücken der ESC-Taste, um das Programm zu beenden
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Schließe die Kamera
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
