# Eyelash Detection & Recommendation API Documentation

## Overview
This API provides intelligent eyelash recommendations based on facial analysis and virtual try-on functionality. It uses MediaPipe Face Mesh for accurate eye detection and classification.

**Base URL:** `http://localhost:5000`

---

## Table of Contents
- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [GET /](#get-)
  - [POST /recommend](#post-recommend)
  - [POST /upload](#post-upload)
  - [POST /adjust](#post-adjust)
  - [GET /settings](#get-settings)
  - [GET /eyelashes](#get-eyelashes)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Available Eyelashes](#available-eyelashes)

---

## Authentication
No authentication required. All endpoints are publicly accessible.

---

## Endpoints

### GET `/`
Get API information and available endpoints.

**Response:**
```json
{
  "message": "Eyelash Detection and Recommendation API",
  "endpoints": {
    "try_on": "/upload (POST) - Try on eyelashes with adjustments",
    "adjust": "/adjust (POST) - Adjust eyelash placement",
    "recommend": "/recommend (POST) - Get eyelash recommendations based on eye analysis",
    "settings": "/settings (GET) - Get default settings and valid ranges",
    "eyelashes": "/eyelashes (GET) - Get all available eyelashes"
  },
  "available_eyelashes": ["Drunk In Love", "Wedding Day", "Foxy", ...]
}
```

---

### POST `/recommend`
Analyze eye characteristics and get personalized eyelash recommendations.

**Request:**
- **Content-Type:** `multipart/form-data`
- **Body Parameters:**
  - `image` (file, required): Image file containing a clear face photo

**Example Request (JavaScript):**
```javascript
const formData = new FormData();
formData.append('image', imageFile);

const response = await fetch('http://localhost:5000/recommend', {
  method: 'POST',
  body: formData
});

const data = await response.json();
```

**Success Response (200):**
```json
{
  "success": true,
  "analysis": {
    "eye_shape": "Almond",
    "eye_size": "Medium",
    "eye_spacing": "Average-set",
    "measurements": {
      "eye_aspect_ratio": 0.45,
      "eye_angle": 1.2,
      "lid_visibility": 0.35,
      "inter_eye_ratio": 1.05,
      "eye_width_to_face_ratio": 0.16,
      "symmetry_score": 0.95
    }
  },
  "recommendations": {
    "top_picks": [
      {
        "name": "Iconic",
        "category": "Doll Eye Styles",
        "style_type": "Universal Doll Eye",
        "description": "Classic doll eye - works for everyone",
        "intensity": "Medium-Heavy",
        "look": "Versatile",
        "match_score": 125
      },
      {
        "name": "Foxy",
        "category": "Cat Eye Styles",
        "style_type": "Elongating & Dramatic",
        "description": "Perfect cat eye with outer corner emphasis",
        "intensity": "Medium",
        "look": "Soft Glam",
        "match_score": 115
      }
    ],
    "all_suitable": [...],
    "total_matches": 6,
    "application_tip": "Apply evenly across entire lash line for balanced look",
    "shape_tip": "Lucky you! Your almond eyes are versatile and can rock any lash style - go bold!"
  }
}
```

**Error Response (400):**
```json
{
  "error": "No face detected in the image"
}
```

---

### POST `/upload`
Apply eyelashes to an image with customizable adjustments.

**Request:**
- **Content-Type:** `multipart/form-data`
- **Body Parameters:**
  - `image` (file, required): Image file containing a clear face photo
  - `choice` (string, required): Eyelash name (see [Available Eyelashes](#available-eyelashes))
  - `vertical_offset` (integer, optional): Vertical position adjustment (-50 to 50, default: -10)
  - `horizontal_offset` (integer, optional): Horizontal position adjustment (-50 to 50, default: 0)
  - `size_scale` (float, optional): Size scale multiplier (0.5 to 3.0, default: 2.0)
  - `height_scale` (float, optional): Height scale multiplier (0.5 to 2.0, default: 1.0)
  - `rotation_angle` (float, optional): Rotation angle in degrees (-45 to 45, default: 0)

**Example Request (JavaScript):**
```javascript
const formData = new FormData();
formData.append('image', imageFile);
formData.append('choice', 'Foxy');
formData.append('vertical_offset', '-15');
formData.append('size_scale', '2.2');
formData.append('rotation_angle', '5');

const response = await fetch('http://localhost:5000/upload', {
  method: 'POST',
  body: formData
});

// Response is a JPEG image
const blob = await response.blob();
const imageUrl = URL.createObjectURL(blob);
```

**Success Response (200):**
- **Content-Type:** `image/jpeg`
- Returns the processed image with eyelashes applied

**Error Response (400):**
```json
{
  "error": "Invalid choice. Must be one of: ['Drunk In Love', 'Wedding Day', ...]"
}
```

---

### POST `/adjust`
Alternative endpoint for applying and adjusting eyelashes (same functionality as `/upload`).

**Request:** Same as `/upload`

**Response:** Same as `/upload`

---

### GET `/settings`
Get default settings and valid ranges for adjustment parameters.

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

### GET `/eyelashes`
Get detailed information about all available eyelashes in inventory.

**Response:**
```json
{
  "success": true,
  "total": 9,
  "eyelashes": [
    {
      "name": "Foxy",
      "category": "Cat Eye Styles",
      "style": "Elongating & Dramatic",
      "intensity": "Medium",
      "look": "Soft Glam",
      "description": "Perfect cat eye with outer corner emphasis",
      "suitable_sizes": ["Small", "Medium", "Large"],
      "suitable_shapes": ["Almond", "Round", "Upturned"]
    },
    {
      "name": "Iconic",
      "category": "Doll Eye Styles",
      "style": "Universal Doll Eye",
      "intensity": "Medium-Heavy",
      "look": "Versatile",
      "description": "Classic doll eye - works for everyone",
      "suitable_sizes": ["Small", "Medium", "Large"],
      "suitable_shapes": ["Round", "Almond", "Upturned", "Downturned", "Hooded"]
    }
  ]
}
```

---

## Data Models

### Eye Analysis
```typescript
interface EyeAnalysis {
  eye_shape: "Hooded" | "Round" | "Almond" | "Upturned" | "Downturned";
  eye_size: "Small" | "Medium" | "Large";
  eye_spacing: "Close-set" | "Average-set" | "Wide-set";
  measurements: {
    eye_aspect_ratio: number;      // Height/width ratio
    eye_angle: number;              // Angle in degrees
    lid_visibility: number;         // Visibility score (0-1)
    inter_eye_ratio: number;        // Inter-eye distance ratio
    eye_width_to_face_ratio: number; // Relative eye size
    symmetry_score: number;         // Symmetry score (0-1)
  };
}
```

### Recommendation
```typescript
interface Recommendation {
  name: string;              // Eyelash product name
  category: string;          // "Cat Eye Styles" | "Doll Eye Styles" | "Natural Styles"
  style_type: string;        // Style description
  description: string;       // Detailed description
  intensity: string;         // "Natural" | "Medium" | "Medium-Heavy" | "Heavy"
  look: string;             // "Natural" | "Glam" | "Soft Glam" | "Dramatic" | "Versatile"
  match_score: number;      // Confidence score (0-200+)
}
```

---

## Error Handling

All endpoints follow a consistent error format:

```json
{
  "error": "Error message describing what went wrong"
}
```

### Common Error Codes:
- **400 Bad Request**: Missing required parameters or invalid input
- **500 Internal Server Error**: Server-side processing error

### Common Error Messages:
- `"No image file provided"` - Image file is missing from request
- `"No face detected in the image"` - No face found in the uploaded image
- `"Invalid choice. Must be one of: [...]"` - Invalid eyelash name provided
- `"Could not load image"` - Image file is corrupted or invalid format
- `"Vertical offset must be between -50 and 50"` - Parameter out of valid range

---

## Available Eyelashes

### Cat Eye Styles
| Name | Suitable Sizes | Best For |
|------|---------------|----------|
| **Foxy** | Small, Medium, Large | All eyes - soft glam look |
| **Drunk In Love** | Small, Medium | Everyday glamour |
| **Other Half 2** | Small | Lightweight cat eye |
| **Vixen** | Large | Dramatic cat eye |

### Doll Eye Styles
| Name | Suitable Sizes | Best For |
|------|---------------|----------|
| **Iconic** | Small, Medium, Large | Universal - works for everyone |
| **Wedding Day** | Small, Medium | Romantic occasions |
| **Staycation** | Large | Dramatic doll effect |

### Natural Styles
| Name | Suitable Sizes | Best For |
|------|---------------|----------|
| **Flare** | Small, Medium | Natural with subtle lift |
| **Other Half 1** | Small | Perfect for hooded eyes |

---

## Image Requirements

### For Best Results:
- **Format**: JPG, JPEG, PNG
- **Resolution**: Minimum 640x480px, recommended 1280x720px or higher
- **Face Position**: Front-facing, well-lit
- **Eyes**: Open, clearly visible
- **Background**: Any (algorithm focuses on face)

### Tips for Users:
1. Take photo in good lighting (natural daylight is best)
2. Look directly at the camera
3. Keep face centered in frame
4. Avoid heavy shadows on face
5. Ensure eyes are fully open and visible

---

## Integration Examples

### React Example
```jsx
import React, { useState } from 'react';

function EyelashRecommender() {
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await fetch('http://localhost:5000/recommend', {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      setRecommendations(data);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input 
        type="file" 
        accept="image/*" 
        onChange={handleImageUpload}
        disabled={loading}
      />
      
      {loading && <p>Analyzing...</p>}
      
      {recommendations && (
        <div>
          <h3>Your Eye Profile</h3>
          <p>Shape: {recommendations.analysis.eye_shape}</p>
          <p>Size: {recommendations.analysis.eye_size}</p>
          
          <h3>Recommended Eyelashes</h3>
          {recommendations.recommendations.top_picks.map((rec, idx) => (
            <div key={idx}>
              <h4>{rec.name}</h4>
              <p>{rec.description}</p>
              <p>Match Score: {rec.match_score}%</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default EyelashRecommender;
```

### Try-On Example
```jsx
function EyelashTryOn() {
  const [resultImage, setResultImage] = useState(null);

  const applyEyelashes = async (imageFile, eyelashChoice) => {
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('choice', eyelashChoice);
    formData.append('vertical_offset', '-10');
    formData.append('size_scale', '2.0');

    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData
      });
      
      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      setResultImage(imageUrl);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div>
      {resultImage && <img src={resultImage} alt="Try-on result" />}
    </div>
  );
}
```

---

## Rate Limiting
Currently, no rate limiting is implemented. For production use, consider implementing rate limiting based on your requirements.

---

## CORS Configuration
CORS is enabled for all origins. For production, configure specific allowed origins in the Flask app:

```python
CORS(app, origins=["https://yourdomain.com"])
```

---

## Support & Issues
For technical support or to report issues, please contact the development team or create an issue in the project repository.

---

## Version
**API Version:** 2.0  
**Last Updated:** November 2025  
**Powered by:** MediaPipe Face Mesh, OpenCV, Flask