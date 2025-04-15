# 👤 Face Recognition API using FastAPI & Dlib

This project is a Face Recognition-based Authentication API built with **FastAPI**, **Dlib**, and **MongoDB**. It allows users to register and log in using **facial recognition via webcam**.

> ⚠️ Pretrained Dlib models are stored on Google Drive. Download links are provided below.
> [Models](https://drive.google.com/drive/folders/1wCzHBQQiQEJuTz7SoGfoIF-oJMZL64R1?usp=sharing)
---

## 📌 Features

- Register using **Email, Password, and Face**
- Login via **Live Face Detection** using Webcam
- **JWT Token** generation for session handling
- Uses **Dlib** for face detection and embedding
- Stores user data securely in **MongoDB Atlas**
- Passwords are hashed using **bcrypt**

---

## 🧠 Tech Stack

- **FastAPI** - Web framework for APIs
- **Dlib** - Face landmark & embedding model
- **OpenCV** - Webcam video stream capture
- **Motor** - Async MongoDB driver
- **PyJWT** - JWT authentication
- **Passlib** - Password hashing
- **Pydantic** - Data validation
- **NumPy / bson** - Array & ID handling

---
## 📚 Libraries Used

**Library	Usage**
- **FastAPI**	- To create the RESTful API backend
- **Motor** -	Async MongoDB driver to store user data
- **Passlib** -	To hash and verify user passwords securely
- **JWT (PyJWT)** -	To generate and validate access tokens
- **Dlib** -	For facial landmark detection and face embeddings
- **OpenCV** -	For accessing the webcam and handling image frames
- **NumPy** -	For numerical operations on face vectors
- **Scipy** - For computing distances between face embeddings

---

## 🚀 API Endpoints

### 🔸 POST `/register`
- **Input**: JSON `{ "email": "", "password": "" }`
- **Action**: Captures face via webcam, stores embedding with credentials
- **Output**: `{"message": "User registered successfully", "face_id": 1234}`

---

### 🔸 POST `/login`
- **Input**: JSON `{ "face_id": 1234 }`
- **Action**: Captures face, compares with stored embedding
- **Output**: `{ "access_token": "JWT_TOKEN" }` if matched

---

### 🔸 GET `/me?token=JWT_TOKEN`
- **Input**: JWT token as query param
- **Output**: `{ "email": "user@example.com" }`

---

## 🧾 Project Structure
- main.py # FastAPI app with routes 

- dlibface.py # Face embedding, registration, and comparison 

 - requirements.txt # Python dependencies 

- README.md # Project documentation

---
## Install dependencies:

~~~cmd
  pip install -r requirements.txt
~~~

To start the FastAPI server, use:
~~~
  uvicorn main:app --reload
~~~

Make sure your webcam is connected and MongoDB URI is configured properly.
