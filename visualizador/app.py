from flask import Flask, render_template, Response
import cv2
import os
import mediapipe as mp

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ========== Detección de Mano ==========
mp_hands = mp.solutions.hands
mp_dibujo = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

def generar_frames():
    cap = cv2.VideoCapture(0)

    overlay1 = cv2.imread('static/uploads/particula.png', cv2.IMREAD_UNCHANGED)
    overlay2 = cv2.imread('static/uploads/particula2.png', cv2.IMREAD_UNCHANGED)
    overlay1 = cv2.resize(overlay1, (100, 100))
    overlay2 = cv2.resize(overlay2, (100, 100))

    pos1 = [100, 100]
    pos2 = [400, 100]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = hands.process(img_rgb)

        if resultado.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(resultado.multi_hand_landmarks):
                mp_dibujo.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape

                # Coordenadas de dedos
                puntos = {id: hand_landmarks.landmark[id] for id in [4, 8, 12, 16, 20, 9]}
                coords = {k: (int(v.x * w), int(v.y * h)) for k, v in puntos.items()}

                dist_thumb_index = ((coords[4][0] - coords[8][0])**2 + (coords[4][1] - coords[8][1])**2)**0.5
                dist_all = sum([
                    ((coords[id][0] - coords[9][0])**2 + (coords[id][1] - coords[9][1])**2)**0.5
                    for id in [4, 8, 12, 16, 20]
                ])

                # Gesto de Mano Cerrada → Mover
                if dist_all < 200:
                    if i == 0:
                        pos1 = [coords[8][0] - 50, coords[8][1] - 50]
                    elif i == 1:
                        pos2 = [coords[8][0] - 50, coords[8][1] - 50]

                # Gesto de Zoom → Solo índice y pulgar
                
               # === Gesto de Zoom: solo índice y pulgar cerca, mano abierta ===
                if dist_thumb_index < 50 and dist_all > 200:
                    base_dist = 50
                    max_scale = 2.0
                    min_scale = 0.5

                    scale = min(max(dist_thumb_index / base_dist, min_scale), max_scale)
                    new_size = int(100 * scale)

                    if i == 0:
                        overlay1 = cv2.imread('static/uploads/particula.png', cv2.IMREAD_UNCHANGED)
                        overlay1 = cv2.resize(overlay1, (new_size, new_size))
                    elif i == 1:
                        overlay2 = cv2.imread('static/uploads/particula2.png', cv2.IMREAD_UNCHANGED)
                        overlay2 = cv2.resize(overlay2, (new_size, new_size))



        def dibujar_overlay(frame, overlay, pos):
            try:
                x, y = pos
                y1, y2 = y, y + overlay.shape[0]
                x1, x2 = x, x + overlay.shape[1]

                if y2 < frame.shape[0] and x2 < frame.shape[1]:
                    roi = frame[y1:y2, x1:x2]
                    mask = overlay[:, :, 3]
                    mask_inv = cv2.bitwise_not(mask)
                    img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                    img_fg = cv2.bitwise_and(overlay[:, :, :3], overlay[:, :, :3], mask=mask)
                    dst = cv2.add(img_bg, img_fg)
                    frame[y1:y2, x1:x2] = dst
            except:
                pass

        dibujar_overlay(frame, overlay1, pos1)
        dibujar_overlay(frame, overlay2, pos2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# ========== Rutas de Flask ==========
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camara')
def camara():
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
