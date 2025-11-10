# 👁️ Eyelash Detection, Placement & Recommendation API

## 📌 Overview
The **Eyelash Detection, Placement & Recommendation API** provides two powerful features:
1. **Virtual Try-On**: Upload an image and automatically apply virtual eyelashes to detected eyes using **MediaPipe FaceMesh** with fine-tuning controls
2. **AI Recommendation System**: Analyzes eye characteristics (size, shape, hooded detection) and recommends the most suitable eyelashes from your catalog

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
Returns information about the API and all available endpoints.

**Request**
```
GET /
```

**Response**
```json
{
  "message": "Eyelash Detection and Recommendation API",
  "endpoints": {
    "try_on": "/upload (POST) - Try on eyelashes with adjustments",
    "adjust": "/adjust (POST) - Adjust eyelash placement",
    "recommend": "/recommend (POST) - Get eyelash recommendations based on eye analysis",
    "settings": "/settings (GET) - Get default settings and valid ranges",
    "eyelashes": "/eyelashes (GET) - Get all available eyelashes"
  }
}
```

---

## 🎨 Virtual Try-On Endpoints

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

## 🤖 AI Recommendation System

### **POST /recommend**
Analyzes a user's facial image to detect eye characteristics and recommends the most suitable eyelashes based on:
- **Eye Size** (Small, Medium, Big)
- **Eye Shape** (Cat eye, Doll eye, etc.)
- **Hooded Eyes Detection**
- **Distance-Invariant Analysis** (consistent results regardless of camera distance)

**Endpoint**
```
POST /recommend
```

**Content-Type:**  
`multipart/form-data`

---

#### 🧾 Form Parameters

| Field | Type | Required | Description |
|--------|------|-----------|-------------|
| `image` | File | ✅ | The face image to analyze |

---

#### ✅ Example Requests

##### cURL
```bash
curl -X POST https://tryon-t0tg.onrender.com/recommend \
  -F "image=@user_photo.jpg"
```

##### Python (requests)
```python
import requests

url = "https://tryon-t0tg.onrender.com/recommend"
files = {"image": open("user_photo.jpg", "rb")}

response = requests.post(url, files=files)
data = response.json()

print(f"Eye Size: {data['analysis']['eye_size']}")
print(f"Is Hooded: {data['analysis']['is_hooded']}")

for rec in data['recommendations']:
    print(f"{rec['name']}: {rec['confidence']}% match - {rec['description']}")
```

##### JavaScript (Fetch)
```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);

fetch('https://tryon-t0tg.onrender.com/recommend', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('Eye Analysis:', data.analysis);
  console.log('Recommendations:', data.recommendations);
});
```

---

#### 📤 Response

**Success Response (200 OK)**
```json
{
  "success": true,
  "analysis": {
    "eye_size": "Medium",
    "is_hooded": false,
    "eye_width": 85.32,
    "eye_height": 25.67,
    "aspect_ratio": 3.32
  },
  "recommendations": [
    {
      "name": "Iconic",
      "style": "Slight Cat Eye",
      "intensity": "Medium Heavy",
      "look": "Versatile",
      "description": "Slight cat eye, medium-heavy, wispy lash style, suitable for all eye shapes.",
      "confidence": 85,
      "image_id": "4"
    },
    {
      "name": "Wedding Day",
      "style": "Doll Eye",
      "intensity": "Medium",
      "look": "Versatile",
      "description": "Doll eye, medium, suitable for all eye shapes and sizes.",
      "confidence": 85,
      "image_id": "9"
    },
    {
      "name": "Foxy",
      "style": "Slight Cat Eye",
      "intensity": "Medium",
      "look": "Soft Glam",
      "description": "Slight Cat eye, medium, super wispy gives a soft cat eye look without looking too heavy. Perfect for medium to big eyes.",
      "confidence": 80,
      "image_id": "3"
    }
  ],
  "total_recommendations": 3
}
```

**Error Response (400)**
```json
{
  "error": "No face detected in the image"
}
```

---

#### 📊 Analysis Fields Explained

| Field | Description |
|-------|-------------|
| `eye_size` | Categorized eye size: "Small", "Medium", "Big", "Hooded Small", or "Hooded Medium" |
| `is_hooded` | Boolean indicating if eyes are hooded (true/false) |
| `eye_width` | Average horizontal eye width in pixels |
| `eye_height` | Average vertical eye height in pixels |
| `aspect_ratio` | Width-to-height ratio of the eyes |

---

#### 🎯 Recommendation Fields Explained

| Field | Description |
|-------|-------------|
| `name` | Eyelash product name |
| `style` | Style category (e.g., "Cat Eye", "Doll Eye", "Half Lash") |
| `intensity` | Lash intensity level (e.g., "Natural", "Medium", "Heavy") |
| `look` | Overall look achieved (e.g., "Glam", "Natural", "Dramatic") |
| `description` | Detailed description of the eyelashes |
| `confidence` | Confidence score (0-100) indicating suitability |
| `image_id` | Corresponding image ID for the eyelash (1-9) |

---

### **GET /eyelashes**
Returns information about all available eyelashes in the catalog.

**Request**
```
GET /eyelashes
```

**Response**
```json
{
  "success": true,
  "total": 9,
  "eyelashes": [
    {
      "name": "Drunk in Love",
      "style": "Cat Eye",
      "intensity": "Medium",
      "look": "Glam",
      "description": "Proper Cat eye lash style, medium (not natural not too heavy) perfect for small hooded eyes if they want a glam look.",
      "image_id": "1",
      "suitable_for": ["Small", "Hooded Small"]
    },
    {
      "name": "Flare",
      "style": "Doll Eye",
      "intensity": "Natural",
      "look": "Natural",
      "description": "Natural doll eye, designed for small hooded eyes. For natural looks.",
      "image_id": "2",
      "suitable_for": ["Small", "Hooded Small"]
    },
    // ... 7 more eyelashes
  ]
}
```

---

## 📋 Available Eyelash Catalog

| Name | Style | Intensity | Best For | Image ID |
|------|-------|-----------|----------|----------|
| **Drunk in Love** | Cat Eye | Medium | Small, Hooded Small | 1 |
| **Flare** | Doll Eye | Natural | Small, Hooded Small | 2 |
| **Foxy** | Slight Cat Eye | Medium | Medium, Big | 3 |
| **Iconic** | Slight Cat Eye | Medium-Heavy | All Sizes | 4 |
| **Other Half 1** | Half Lash Natural | Natural | All Sizes | 5 |
| **Other Half 2** | Half Lash Cat | Medium | All Sizes | 6 |
| **Staycation** | Doll Eye | Heavy | Medium, Big (Not Hooded) | 7 |
| **Vixen** | Cat Eye | Heavy | Big, Hooded Medium | 8 |
| **Wedding Day** | Doll Eye | Medium | All Sizes | 9 |

---

## 🔄 Typical Workflow

### 1️⃣ Get Recommendations First
```bash
# Step 1: Analyze eyes and get recommendations
curl -X POST https://tryon-t0tg.onrender.com/recommend \
  -F "image=@user_photo.jpg" \
  -o recommendations.json
```

### 2️⃣ Try On Recommended Eyelashes
```bash
# Step 2: Try on the top recommended eyelash
curl -X POST https://tryon-t0tg.onrender.com/upload \
  -F "image=@user_photo.jpg" \
  -F "choice=4" \
  --output tryon_result.jpg
```

### 3️⃣ Fine-Tune Placement (Optional)
```bash
# Step 3: Adjust if needed
curl -X POST https://tryon-t0tg.onrender.com/adjust \
  -F "image=@user_photo.jpg" \
  -F "choice=4" \
  -F "vertical_offset=-15" \
  -F "size_scale=2.2" \
  --output final_result.jpg
```

---

## 🧪 Key Features of the Recommendation System

### ✅ Distance-Invariant Analysis
The recommendation system uses **normalized facial proportions** rather than absolute pixel measurements. This means:
- A person at 1 meter distance will get the same recommendations as at 2 meters
- Consistent results across different image resolutions
- Reliable for both selfies and professional photos

### ✅ Hooded Eye Detection
Advanced algorithm detects hooded eyes by analyzing:
- Eyelid-to-eyebrow distance
- Upper eyelid visibility
- Eye opening characteristics

### ✅ Multi-Factor Matching
Recommendations are based on:
- **Eye Size**: Small, Medium, Big
- **Eye Shape**: Aspect ratio and proportions
- **Hooded Classification**: Special consideration for hooded eyes
- **Lash Characteristics**: Style, intensity, and look

### ✅ Confidence Scoring
Each recommendation includes a confidence score (0-100%) indicating:
- How well the eyelash matches the detected eye characteristics
- Specificity of the recommendation
- Versatility of the eyelash style

---

## ⚠️ Error Handling

### Try-On Endpoints (`/upload`, `/adjust`)

| Error Type | HTTP Code | Example Message |
|-------------|------------|----------------|
| Missing Image | 400 | `"error": "No image file provided"` |
| Missing Choice | 400 | `"error": "No eyelash choice provided"` |
| Invalid Choice | 400 | `"error": "Invalid choice. Must be 1, 2, 3, or 4."` |
| Invalid Range | 400 | `"error": "Vertical offset must be between -50 and 50"` |
| No Face Detected | 500 | `"error": "No face detected"` |
| Missing Eyelash File | 500 | `"error": "Eyelash image not found"` |

### Recommendation Endpoint (`/recommend`)

| Error Type | HTTP Code | Example Message |
|-------------|------------|----------------|
| Missing Image | 400 | `"error": "No image file provided"` |
| No Face Detected | 400 | `"error": "No face detected in the image"` |
| Invalid Image | 400 | `"error": "Could not read image"` |
| Processing Error | 500 | `"error": "An error occurred: [details]"` |

---

## 💡 Best Practices

### For Frontend Integration

1. **Show Loading States**: Image analysis can take 2-3 seconds
2. **Handle Both Success and Error**: Always check the `success` field
3. **Display Confidence Scores**: Help users understand recommendation strength
4. **Allow Multiple Tries**: Let users try all recommended options
5. **Progressive Enhancement**: Show basic recommendations first, load images after

### For Image Quality

1. **Face Visibility**: Ensure the face is clearly visible and well-lit
2. **Resolution**: Minimum 480x640 pixels recommended
3. **Format**: JPEG, PNG supported
4. **File Size**: Keep under 5MB for optimal performance
5. **Orientation**: Front-facing photos work best

### For Optimal Results

1. **Use `/recommend` First**: Get AI-powered suggestions before trying on
2. **Try Top 3 Recommendations**: Test the highest confidence options
3. **Fine-Tune with `/adjust`**: Use adjustment parameters for perfect placement
4. **Consider Eye Type**: Pay attention to hooded eye recommendations

---

## 🔧 Technical Details

### MediaPipe Face Mesh
- **468 facial landmarks** detected
- **High accuracy** eye region detection
- **Real-time processing** capability
- **Refined landmarks** for precise placement

### Recommendation Algorithm
- **Normalized measurements** for distance invariance
- **Multiple feature analysis** (width, height, aspect ratio)
- **Hooded eye detection** using eyelid-to-eyebrow ratios
- **Confidence scoring** based on multi-factor matching

---

## 📞 Support

For issues, questions, or feature requests, please contact the development team or create an issue in the project repository.

---

**API Version**: 2.0  
**Last Updated**: 2025  
**Powered by**: MediaPipe, Flask, OpenCV
