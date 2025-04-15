from fastapi import FastAPI
from pydantic import BaseModel
import dlibface
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
import jwt
from jwt.exceptions import InvalidTokenError
from bson.objectid import ObjectId
import numpy as np

MONGO_URI = "mongodb+srv://beingdillig:4CNlIcaPviSu0c72@todo.sbvg3.mongodb.net/?retryWrites=true&w=majority&appName=todo"


# Local module (your code)
import dlibface

app = FastAPI()

# MongoDB Connection
client = AsyncIOMotorClient(MONGO_URI)
db = client["face_auth"]
user_collection = db["users"]

# Password hasher
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT secret and settings
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Pydantic Model
class UserRegister(BaseModel):
    email: str
    password: str

class FaceLogin(BaseModel):
    face_id: str

# Helper Functions
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def get_password_hash(password):
    return pwd_context.hash(password)

# 📝 Registration Route
@app.post("/register")
async def register_user(user: UserRegister):
    existing_user = await user_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    face_embedding = dlibface.get_face_embedding()
    if face_embedding is None:
        raise HTTPException(status_code=400, detail="Face not detected")

    total_users = await user_collection.count_documents({})
    face_id = str(total_users).zfill(4)
    embedding_list = face_embedding.tolist()

    hashed_pw = get_password_hash(user.password)

    await user_collection.insert_one({
        "email": user.email,
        "hashed_password": hashed_pw,
        "face_id": face_id,
        "face_embedding": embedding_list
    })

    return {"message": "User registered successfully", "face_id": face_id}

# 🔐 Face Login Route
@app.post("/login")
async def face_login(data: FaceLogin):
    user = await user_collection.find_one({"face_id": data.face_id})
    if not user:
        raise HTTPException(status_code=404, detail="Face ID not found")

    captured_embedding = dlibface.get_face_embedding(frames_count=1)
    if captured_embedding is None:
        raise HTTPException(status_code=400, detail="Face not detected")

    saved_embedding = np.array(user["face_embedding"])
    new_embedding = np.array(captured_embedding)
    dist = np.linalg.norm(saved_embedding - new_embedding)

    if dist < 0.45:
        access_token = create_access_token(data={"sub": user["email"]})
        return {"access_token": access_token, "token_type": "bearer"}
    else:
        raise HTTPException(status_code=401, detail="Face verification failed")

# 🎯 Optional route to verify JWT
@app.get("/me")
async def get_current_user(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"email": payload.get("sub")}
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
