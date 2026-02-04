import cv2
import numpy as np
import uuid
from insightface.app import FaceAnalysis
from .database import conn

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

THRESHOLD = 0.6


def get_embedding(image_bytes):
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    faces = app.get(img)
    if not faces:
        return None

    return faces[0].embedding


def find_match(embedding):
    cur = conn.cursor()
    cur.execute("""
        SELECT user_id, embedding <-> %s AS distance
        FROM face_embeddings
        ORDER BY distance
        LIMIT 1;
    """, (embedding.tolist(),))

    row = cur.fetchone()
    if row and row[1] < THRESHOLD:
        return row[0], row[1]

    return None, None


def register_user(name, embedding):
    user_id = str(uuid.uuid4())
    emb_id = str(uuid.uuid4())

    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (id, name) VALUES (%s, %s)",
        (user_id, name)
    )
    cur.execute(
        "INSERT INTO face_embeddings (id, user_id, embedding) VALUES (%s, %s, %s)",
        (emb_id, user_id, embedding.tolist())
    )

    conn.commit()
    return user_id
