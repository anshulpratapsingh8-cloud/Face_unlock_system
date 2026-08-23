from flask import Flask, request, jsonify
import os
from deepface import DeepFace

app = Flask(__name__)

# 📁 dataset folder create
if not os.path.exists("dataset"):
    os.makedirs("dataset")


# ➕ REGISTER FACE
@app.route('/register', methods=['POST'])
def register():
    try:
        file = request.files['image']

        save_path = os.path.join("dataset", "user.jpg")
        file.save(save_path)

        return jsonify({"status": "face added successfully"})

    except Exception as e:
        print("REGISTER ERROR:", e)
        return jsonify({"status": "error"})


# 🔓 UNLOCK FACE
@app.route('/unlock', methods=['POST'])
def unlock():
    try:
        file = request.files['image']

        temp_path = "temp.jpg"
        file.save(temp_path)

        # 🔍 DeepFace match
        result = DeepFace.find(
            img_path=temp_path,
            db_path="dataset",
            enforce_detection=False,
            model_name="Facenet"
        )

        # 🧹 temp delete
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if len(result) > 0 and len(result[0]) > 0:
            distance = result[0].iloc[0]['distance']
            print("Distance:", distance)

            if distance < 0.4:
                return jsonify({"status": "unlocked"})
            else:
                return jsonify({"status": "face not matched"})

        return jsonify({"status": "no face found"})

    except Exception as e:
        print("UNLOCK ERROR:", e)
        return jsonify({"status": "error"})


# ▶️ RUN SERVER
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)