import pandas as pd
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify,send_file
import cv2
import imutils
import numpy as np
import pickle
import sqlite3
import base64
from datetime import datetime
import os
import sys
import subprocess
import time
import pandas as pd
import io
import openpyxl
app = Flask(__name__)
protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
embedder = cv2.dnn.readNetFromTorch("openface/openface_nn4.small2.v1.t7")
recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())
PROB_THRESHOLD = 0.80
recognized_employee = None
url1="https://192.168.0.138:8080/video"
url=0

def get_db_connection():
    conn = sqlite3.connect('database/employees.db')
    conn.row_factory = sqlite3.Row
    return conn
def sql_to_df():
    conn = sqlite3.connect('database/employees.db')
    query = "SELECT id,first_name,last_name, position, phone, date, presence from Absence"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def generate_frames1():
    cap = cv2.VideoCapture(url)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_frames():
    global recognized_employee
    current_employee = None
    vs = cv2.VideoCapture(url)
    vs.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        success, frame = vs.read()
        if not success:
            break
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                                         (104.0, 177.0, 123.0), swapRB=False, crop=False)
        detector.setInput(imageBlob)
        detections = detector.forward()

        detected = False
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                if fW < 20 or fH < 20:
                    continue
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]
                if proba >= PROB_THRESHOLD:
                    text = "{}: {:.2f}%".format(name, proba * 100)
                    info = get_employee_info(name)
                    if info:
                        recognized_employee = info
                        recognized_employee['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        _, buffer = cv2.imencode('.jpg', face)
                        recognized_employee['photo'] = base64.b64encode(buffer).decode('utf-8')
                        detected = True
                        current_employee = name
                        insert_absence_record(recognized_employee)
                else:
                    text = "Unknown"
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video_feed1')
def video_feed1():
    return Response(generate_frames1(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/capture_photos', methods=['POST'])
def capture_photos_route():
    first_name = request.form['first_name']
    last_name = request.form['last_name']
    capture_photos(first_name, last_name)
    return render_template('train.html', first_name=first_name, last_name=last_name)

def capture_photos(first_name, last_name, num_photos=30):
    output_dir = f'dataset/{first_name.lower()}-{last_name.lower()}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Erreur : La caméra n'a pas pu être ouverte.")
        return
    cam.set(3, 640)
    cam.set(4, 480)

    count = 0
    while count < num_photos:
        ret, frame = cam.read()
        if not ret:
            print("Erreur : Impossible de lire l'image de la caméra.")
            break

        file_name = f"{count:04d}.jpg"
        file_path = os.path.join(output_dir, file_name)
        cv2.imwrite(file_path, frame)
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

@app.route('/train_model', methods=['POST'])
def train_model():
    global recognizer
    global le
    try:
        python_executable = sys.executable
        result = subprocess.run([python_executable, 'extract_embeddings.py'], check=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        print(result.stdout)
        print(result.stderr)
        result = subprocess.run([python_executable, 'train_model.py'], check=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        print(result.stdout)
        print(result.stderr)
        recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
        le = pickle.loads(open("output/le.pickle", "rb").read())

        return redirect(url_for('index'))
    except subprocess.CalledProcessError as e:
        return f"Une erreur s'est produite: {e}", 500
@app.route('/capture')
def index():
    return render_template('capture_photo.html')

@app.route('/adding', methods=['GET', 'POST'])
def adding_employee():
    if request.method == 'POST':
        first_name = request.form['firstName']
        last_name = request.form['lastName']
        position = request.form['position']
        start_date = request.form['startDate']
        phone = request.form['phone']
        photo = request.files['photo'].read() if 'photo' in request.files else None
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO employees (first_name, last_name, position, start_date, phone, photo)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (first_name, last_name, position, start_date, phone, photo))
        conn.commit()
        conn.close()
        return redirect(url_for('adding_employee'))

    conn = get_db_connection()
    employees = conn.execute('SELECT * FROM employees').fetchall()
    conn.close()
    employees = [{
        'first_name': emp['first_name'],
        'last_name': emp['last_name'],
        'position': emp['position'],
        'start_date': emp['start_date'],
        'phone': emp['phone'],
        'photo': base64.b64encode(emp['photo']).decode('utf-8') if emp['photo'] else None
    } for emp in employees]
    return render_template('add_employee.html', employees=employees)

@app.route('/face_reco')
def face_reco():
    return render_template('face_reco.html')

@app.route('/employee_info')
def employee_info():
    return jsonify(recognized_employee or {})

@app.route('/show_absence')
def show_absence():
    try:
        conn = sqlite3.connect("database/employees.db")
        cur = conn.cursor()
        cur.execute("SELECT * FROM Absence")
        rows = cur.fetchall()
        conn.close()
        absence_data = [{
            'id': row[0],
            'photo': row[1],
            'first_name': row[2],
            'last_name': row[3],
            'position': row[4],
            'phone': row[5],
            'date': row[6],
            'presence': row[7]
        } for row in rows]
        return jsonify(absence_data)
    except Exception as e:
        print(f"Error in show_absence: {e}")
        return jsonify({"error": "An error occurred while retrieving data."})

@app.route('/view_absence')
def view_absence():
    return render_template('absence_table.html')

def get_employee_info(name):
    try:
        NM = name.split('-')
        first_name, last_name = NM
        conn = sqlite3.connect("database/employees.db")
        cur = conn.cursor()
        cur.execute("SELECT id, first_name, last_name, position, phone, photo FROM employees WHERE first_name = ? AND last_name = ?", (first_name, last_name))
        result = cur.fetchone()
        conn.close()
        if result:
            return {
                'id': result[0],
                'first_name': result[1],
                'last_name': result[2],
                'position': result[3],
                'phone': result[4],
                'photo': base64.b64encode(result[5]).decode('utf-8') if result[5] else None
            }
    except Exception as e:
        print(f"Error in get_employee_info: {e}")
        return None


def insert_absence_record(employee):
    try:
        conn = sqlite3.connect("database/employees.db")
        cur = conn.cursor()


        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M")

        cur.execute("SELECT * FROM Absence WHERE first_name = ? AND last_name = ? AND date(date) = ?",
                    (employee['first_name'], employee['last_name'], current_date))
        existing_record = cur.fetchone()

        if not existing_record:
            presence_status = 'P' if "08:00" <= current_time <= "08:15" else 'A'

            cur.execute(
                "INSERT INTO Absence (photo, first_name, last_name, position, phone, date, presence) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (employee['photo'], employee['first_name'], employee['last_name'], employee['position'],
                 employee['phone'], current_date, current_time))
            conn.commit()

    except sqlite3.Error as e:
        print(f"Erreur lors de l'insertion des données : {e}")
    finally:
        conn.close()
@app.route('/download_excel')
def download_excel():
    df = sql_to_df()
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')

    output.seek(0)

    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='data.xlsx'
    )
@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def to_login():
    email = request.form['email']
    password = request.form['password']
    if email == 'hassannfissi27@gmail.com' and password == 'Hassan':
        return redirect(url_for('home'))
    else:
        return 'Identifiant invalide'

@app.route('/home')
def home():
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)