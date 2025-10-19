# 👁️ Eyelash Detection API

## 📌 Overview
The **Eyelash Detection API** allows users to upload an **image** along with a **choice number (1–4)** that represents an eyelash image selection.  
If both the image and choice are provided correctly, the API responds with `"Input successful"`.

---

## 🌐 Base URL
**Production URL:**  
```
https://tryon-t0tg.onrender.com
```

---

## 🚀 Endpoints

### **GET /**
Returns a simple message confirming that the API is active.

**Request**
```
GET /
```

**Response**
```json
{
  "message": "Eyelash Detection API"
}
```

---

### **POST /upload**
Uploads an image file and a choice number (1–4).  
Validates both inputs and returns a success message if correct.

**Endpoint**
```
POST /upload
```

**Content-Type:**  
`multipart/form-data`

**Form Parameters**

| Field | Type | Required | Description |
|--------|------|-----------|-------------|
| `image` | File | ✅ | The image file to upload |
| `choice` | Integer/String | ✅ | Eyelash option number (1, 2, 3, or 4) |

---

## ✅ Example Requests

### cURL
```bash
curl -X POST https://tryon-t0tg.onrender.com/upload   -F "image=@sample.jpg"   -F "choice=2"
```

### Python (requests)
```python
import requests

url = "https://tryon-t0tg.onrender.com/upload"
files = {"image": open("sample.jpg", "rb")}
data = {"choice": "2"}

response = requests.post(url, files=files, data=data)
print(response.json())
```

---

## 🧩 Example Responses

### ✅ Success (HTTP 200)
```json
{
  "message": "Input successful",
  "choice": "2",
  "filename": "sample.jpg"
}
```

### ❌ Error: No image file
```json
{
  "error": "No image file found"
}
```

### ❌ Error: No choice provided
```json
{
  "error": "No choice provided"
}
```

### ❌ Error: Invalid choice
```json
{
  "error": "Invalid choice. Must be 1, 2, 3, or 4."
}
```

---

## ⚙️ Technical Details
- **Framework:** Flask  
- **Runtime:** Python 3.10+  
- **Deployment:** Render  
- **Method:** POST (for uploads)  
- **Upload Type:** multipart/form-data  
- **Port:** 5000 (managed by Render)

---

## 🧪 Local Testing

1. Install dependencies:
   ```bash
   pip install Flask gunicorn
   ```
2. Run the server:
   ```bash
   python app.py
   ```
3. Send a test request:
   ```bash
   curl -X POST http://127.0.0.1:5000/upload -F "image=@test.jpg" -F "choice=3"
   ```

**Expected Response**
```json
{
  "message": "Input successful",
  "choice": "3",
  "filename": "test.jpg"
}
```

---

## 📄 Response Schema
```json
{
  "message": "Input successful",
  "choice": "<string: 1-4>",
  "filename": "<string>"
}
```

---

## 👨‍💻 Developer Notes
- The uploaded image is currently processed in memory (not saved permanently).  
- To save images, uncomment the following line in `app.py`:
  ```python
  image.save(f"uploads/{image.filename}")
  ```
- The `/upload` endpoint is ready for future integration with eyelash overlay or image processing logic.

---

**Author:** Umair Khalid  
**API URL:** [https://tryon-t0tg.onrender.com](https://tryon-t0tg.onrender.com)
