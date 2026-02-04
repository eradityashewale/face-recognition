from fastapi import FastAPI, UploadFile, File, Form
from .face_service import get_embedding, find_match, register_user

app = FastAPI()


@app.post("/identify")
async def identify_face(file: UploadFile = File(...)):
    image_bytes = await file.read()
    embedding = get_embedding(image_bytes)

    if embedding is None:
        return {"success": False, "message": "No face detected"}

    user_id, distance = find_match(embedding)

    if user_id:
        return {
            "success": True,
            "matched": True,
            "user_id": user_id,
            "confidence": round(1 - distance, 2)
        }

    return {
        "success": True,
        "matched": False,
        "message": "User not found"
    }


@app.post("/register")
async def register_face(
    name: str = Form(...),
    file: UploadFile = File(...)
):
    image_bytes = await file.read()
    embedding = get_embedding(image_bytes)

    if embedding is None:
        return {"success": False, "message": "No face detected"}

    user_id = register_user(name, embedding)

    return {
        "success": True,
        "user_id": user_id,
        "message": "User registered successfully"
    }
