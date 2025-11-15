# Eyelash Try-On & Recommendation API Documentation

## Base URL
```
https://tryon-t0tg.onrender.com
```

## Endpoints

### 1. Get API Info
**GET** `/`

Returns API information and available endpoints.

**Response:**
```json
{
  "message": "Eyelash Detection and Recommendation API",
  "endpoints": { ... },
  "available_eyelashes": [...]
}
```

---

### 2. Try-On Eyelashes
**POST** `/upload`

Apply eyelashes to an uploaded image with optional adjustment parameters.

**Request:**
- **Content-Type:** `multipart/form-data`
- **Parameters:**

| Parameter | Type | Required | Default | Range | Description |
|-----------|------|----------|---------|-------|-------------|
| `image` | file | Yes | - | - | User's image file |
| `choice` | string | Yes | - | - | Eyelash name (see available options below) |
| `vertical_offset` | integer | No | -10 | -50 to 50 | Vertical position adjustment |
| `horizontal_offset` | integer | No | 0 | -50 to 50 | Horizontal position adjustment |
| `size_scale` | float | No | 2.0 | 0.5 to 3.0 | Overall size adjustment |
| `height_scale` | float | No | 1.0 | 0.5 to 2.0 | Height-specific scaling |
| `rotation_angle` | float | No | 0 | -45 to 45 | Rotation in degrees |

**Available Eyelash Choices:**
- `Drunk In Love`
- `Wedding Day`
- `Foxy`
- `Flare`
- `Vixen`
- `Other Half 1`
- `Other Half 2`
- `Staycation`
- `Iconic`

**Response:**
- **Content-Type:** `image/jpeg`
- Returns processed image with eyelashes applied

**Error Responses:**
- `400` - Missing required parameters or invalid values
- `500` - Processing error

---

### 3. Adjust Eyelashes
**POST** `/adjust`

Alternative endpoint for applying eyelashes with adjustments (same functionality as `/upload`).

**Request:** Same as `/upload` endpoint

**Response:** Same as `/upload` endpoint

---

### 4. Get Eyelash Recommendations
**POST** `/recommend`

Analyze eye characteristics and receive personalized eyelash recommendations.

**Request:**
- **Content-Type:** `multipart/form-data`
- **Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | file | Yes | User's image file |

**Response:**
```json
{
  "success": true,
  "analysis": {
    "eye_size": "Medium",
    "is_hooded": false,
    "eye_width": 123.45,
    "eye_height": 45.67,
    "aspect_ratio": 2.7
  },
  "recommendations": [
    {
      "name": "Iconic",
      "style": "Slight Cat Eye",
      "intensity": "Medium Heavy",
      "look": "Versatile",
      "description": "Slight cat eye, medium-heavy...",
      "confidence": 85.0,
      "image_id": "4"
    }
  ],
  "total_recommendations": 5
}
```

**Error Responses:**
- `400` - Missing image or no face detected
- `500` - Processing error

---

### 5. Get Default Settings
**GET** `/settings`

Retrieve default adjustment values and valid parameter ranges.

**Response:**
```json
{
  "default_settings": {
    "vertical_offset": -10,
    "horizontal_offset": 0,
    "size_scale": 2.0,
    "height_scale": 1.0,
    "rotation_angle": 0
  },
  "valid_ranges": {
    "vertical_offset": {"min": -50, "max": 50},
    "horizontal_offset": {"min": -50, "max": 50},
    "size_scale": {"min": 0.5, "max": 3.0},
    "height_scale": {"min": 0.5, "max": 2.0},
    "rotation_angle": {"min": -45, "max": 45}
  }
}
```

---

### 6. Get All Eyelashes
**GET** `/eyelashes`

Retrieve information about all available eyelash styles.

**Response:**
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
      "description": "Proper Cat eye lash style...",
      "image_id": "1",
      "suitable_for": ["Small", "Hooded Small"]
    }
  ]
}
```

---

## Error Handling

All endpoints return errors in the following format:
```json
{
  "error": "Error description"
}
```

**Common HTTP Status Codes:**
- `200` - Success
- `400` - Bad Request (invalid parameters)
- `500` - Internal Server Error

---

## Example Usage

### Try-On with Adjustments (cURL)
```bash
curl -X POST http://localhost:5000/upload \
  -F "image=@photo.jpg" \
  -F "choice=Iconic" \
  -F "vertical_offset=-15" \
  -F "size_scale=2.2" \
  --output result.jpg
```

### Get Recommendations (cURL)
```bash
curl -X POST http://localhost:5000/recommend \
  -F "image=@photo.jpg"
```