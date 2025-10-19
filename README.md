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

## 📄 Response Schema
```json
{
  "message": "Input successful",
  "choice": "<string: 1-4>",
  "filename": "<string>"
}
```
