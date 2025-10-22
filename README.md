# 👁️ Eyelash Detection & Placement API

## 📌 Overview
The **Eyelash Detection & Placement API** allows users to upload an image and automatically apply virtual eyelashes to detected eyes using **MediaPipe FaceMesh**.  
You can also fine-tune the eyelash placement using optional adjustment parameters such as size, height, and position offsets.

This API supports **CORS (Cross-Origin Resource Sharing)** — all origins are allowed by default, enabling easy integration with frontend applications.

---

## 🌐 Base URL
**Production URL:**  
```
https://tryon-t0tg.onrender.com
```

---

## 🚀 Endpoints

### **GET /**
Returns a simple message confirming that the API is running.

**Request**
```
GET /
```

**Response**
```json
{
  "message": "Eyelash Detection and Placement API"
}
```

---

### **POST /upload**
Uploads an image file and a selected eyelash style (`choice`) to apply virtual eyelashes.  
You can also provide optional **adjustment parameters** to control the eyelash position and size.

**Endpoint**
```
POST /upload
```

**Content-Type:**  
`multipart/form-data`

---

#### 🧾 Form Parameters

| Field | Type | Required | Description |
|--------|------|-----------|-------------|
| `image` | File | ✅ | The face image to process |
| `choice` | String (1–4) | ✅ | Eyelash style selection |
| `vertical_offset` | Integer | ❌ | Moves lashes up/down (default: `-10`, range: `-50` to `50`) |
| `horizontal_offset` | Integer | ❌ | Moves lashes left/right (default: `0`, range: `-50` to `50`) |
| `size_scale` | Float | ❌ | Scales overall eyelash size (default: `2.0`, range: `0.5`–`3.0`) |
| `height_scale` | Float | ❌ | Scales eyelash height (default: `1.0`, range: `0.5`–`2.0`) |

---

#### ✅ Example Requests

##### cURL
```bash
curl -X POST https://tryon-t0tg.onrender.com/upload \
  -F "image=@sample.jpg" \
  -F "choice=2" \
  -F "vertical_offset=-8" \
  -F "horizontal_offset=5" \
  -F "size_scale=1.8" \
  -F "height_scale=1.1" \
  --output result.jpg
```

##### Python (requests)
```python
import requests

url = "https://tryon-t0tg.onrender.com/upload"
files = {"image": open("sample.jpg", "rb")}
data = {
    "choice": "2",
    "vertical_offset": "-8",
    "horizontal_offset": "5",
    "size_scale": "1.8",
    "height_scale": "1.1"
}

response = requests.post(url, files=files, data=data)
with open("result.jpg", "wb") as f:
    f.write(response.content)
```

---

#### 📤 Response
If successful, the API returns an **image (JPEG)** with eyelashes applied.

**Headers**
```
Content-Type: image/jpeg
Content-Disposition: inline; filename=result.jpg
```

**Error Example**
```json
{
  "error": "No face detected"
}
```

---

### **POST /adjust**
This endpoint works the same way as `/upload`, but is specifically designed for **fine-tuning existing results** by adjusting eyelash position and size interactively.

**Endpoint**
```
POST /adjust
```

**Content-Type:**  
`multipart/form-data`

**Form Parameters:** *(Same as `/upload`)*  
- `image`  
- `choice`  
- `vertical_offset`  
- `horizontal_offset`  
- `size_scale`  
- `height_scale`

---

#### Example Request
```bash
curl -X POST https://tryon-t0tg.onrender.com/adjust \
  -F "image=@sample.jpg" \
  -F "choice=1" \
  -F "vertical_offset=-12" \
  -F "size_scale=2.2" \
  --output adjusted_result.jpg
```

---

### **GET /settings**
Fetches the **default** and **valid range** values for all adjustable parameters.  
This helps frontend developers build dynamic UI sliders or controls for fine-tuning eyelash placement.

**Request**
```
GET /settings
```

**Response**
```json
{
  "default_settings": {
    "vertical_offset": -10,
    "horizontal_offset": 0,
    "size_scale": 2.0,
    "height_scale": 1.0
  },
  "valid_ranges": {
    "vertical_offset": {"min": -50, "max": 50},
    "horizontal_offset": {"min": -50, "max": 50},
    "size_scale": {"min": 0.5, "max": 3.0},
    "height_scale": {"min": 0.5, "max": 2.0}
  }
}
```

---

## ⚠️ Error Handling

| Error Type | HTTP Code | Example Message |
|-------------|------------|----------------|
| Missing Image | 400 | `"error": "No image file provided"` |
| Missing Choice | 400 | `"error": "No eyelash choice provided"` |
| Invalid Choice | 400 | `"error": "Invalid choice. Must be 1, 2, 3, or 4."` |
| Invalid Range | 400 | `"error": "Vertical offset must be between -50 and 50"` |
| No Face Detected | 500 | `"error": "No face detected"` |
| Missing Eyelash File | 500 | `"error": "Eyelash image not found"` |

---