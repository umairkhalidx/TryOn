// Eyelash Recommendation System
class EyelashRecommendationSystem {
    constructor() {
        // Eye landmarks indices
        this.RIGHT_EYE = [33, 133, 160, 159, 158, 157, 173, 155, 154, 153, 145, 144, 163, 7];
        this.LEFT_EYE = [263, 362, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373, 390, 249];
        
        // Key points for measurements
        this.RIGHT_EYE_INNER = 133;
        this.RIGHT_EYE_OUTER = 33;
        this.LEFT_EYE_INNER = 362;
        this.LEFT_EYE_OUTER = 263;
        
        // Inventory with detailed specifications
        this.inventory = {
            "Cat Eye Styles": {
                "Foxy": {
                    "suitable_sizes": ["Small", "Medium", "Large"],
                    "suitable_shapes": ["Almond", "Round", "Upturned"],
                    "style_type": "Elongating & Dramatic",
                    "description": "Perfect cat eye with outer corner emphasis",
                    "intensity": "Medium",
                    "look": "Soft Glam"
                },
                "Drunk In Love": {
                    "suitable_sizes": ["Small", "Medium"],
                    "suitable_shapes": ["Almond", "Round", "Upturned", "Downturned"],
                    "style_type": "Subtle Cat Eye",
                    "description": "Soft cat eye effect for everyday glamour",
                    "intensity": "Medium",
                    "look": "Glam"
                },
                "Other Half 2": {
                    "suitable_sizes": ["Small"],
                    "suitable_shapes": ["Almond", "Round", "Upturned", "Hooded"],
                    "style_type": "Delicate Cat Eye",
                    "description": "Lightweight cat eye for smaller eyes",
                    "intensity": "Medium",
                    "look": "Lifted"
                },
                "Vixen": {
                    "suitable_sizes": ["Large"],
                    "suitable_shapes": ["Almond", "Round", "Upturned"],
                    "style_type": "Bold Cat Eye",
                    "description": "Dramatic cat eye for larger eyes",
                    "intensity": "Heavy",
                    "look": "Dramatic"
                }
            },
            "Doll Eye Styles": {
                "Iconic": {
                    "suitable_sizes": ["Small", "Medium", "Large"],
                    "suitable_shapes": ["Round", "Almond", "Upturned", "Downturned", "Hooded"],
                    "style_type": "Universal Doll Eye",
                    "description": "Classic doll eye - works for everyone",
                    "intensity": "Medium-Heavy",
                    "look": "Versatile"
                },
                "Wedding Day": {
                    "suitable_sizes": ["Small", "Medium"],
                    "suitable_shapes": ["Round", "Almond", "Upturned", "Downturned", "Hooded"],
                    "style_type": "Romantic Doll Eye",
                    "description": "Soft, romantic doll effect",
                    "intensity": "Medium",
                    "look": "Versatile"
                },
                "Staycation": {
                    "suitable_sizes": ["Large"],
                    "suitable_shapes": ["Round", "Almond", "Upturned"],
                    "style_type": "Voluminous Doll Eye",
                    "description": "Full, dramatic doll eye for larger eyes",
                    "intensity": "Heavy",
                    "look": "Dramatic"
                }
            },
            "Natural Styles": {
                "Flare": {
                    "suitable_sizes": ["Small", "Medium"],
                    "suitable_shapes": ["Almond", "Upturned", "Downturned", "Round"],
                    "style_type": "Natural with Subtle Flare",
                    "description": "Natural look with gentle outer corner lift",
                    "intensity": "Natural",
                    "look": "Natural"
                },
                "Other Half 1": {
                    "suitable_sizes": ["Small"],
                    "suitable_shapes": ["Hooded"],
                    "style_type": "Natural for Hooded Eyes",
                    "description": "Specially designed for small hooded eyes",
                    "intensity": "Natural",
                    "look": "Natural"
                }
            }
        };
    }
    
    calculateDistance(point1, point2) {
        return Math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2);
    }
    
    getEyeMeasurements(landmarks, imgWidth, imgHeight) {
        const measurements = {};
        
        // Convert landmarks to pixel coordinates
        const rightEyeInner = [landmarks[this.RIGHT_EYE_INNER].x * imgWidth, landmarks[this.RIGHT_EYE_INNER].y * imgHeight];
        const rightEyeOuter = [landmarks[this.RIGHT_EYE_OUTER].x * imgWidth, landmarks[this.RIGHT_EYE_OUTER].y * imgHeight];
        const leftEyeInner = [landmarks[this.LEFT_EYE_INNER].x * imgWidth, landmarks[this.LEFT_EYE_INNER].y * imgHeight];
        const leftEyeOuter = [landmarks[this.LEFT_EYE_OUTER].x * imgWidth, landmarks[this.LEFT_EYE_OUTER].y * imgHeight];
        
        // Right eye measurements
        const rightEyeTop = [landmarks[159].x * imgWidth, landmarks[159].y * imgHeight];
        const rightEyeBottom = [landmarks[145].x * imgWidth, landmarks[145].y * imgHeight];
        const rightEyeTop2 = [landmarks[160].x * imgWidth, landmarks[160].y * imgHeight];
        const rightEyeBottom2 = [landmarks[144].x * imgWidth, landmarks[144].y * imgHeight];
        
        // Left eye measurements
        const leftEyeTop = [landmarks[386].x * imgWidth, landmarks[386].y * imgHeight];
        const leftEyeBottom = [landmarks[374].x * imgWidth, landmarks[374].y * imgHeight];
        const leftEyeTop2 = [landmarks[387].x * imgWidth, landmarks[387].y * imgHeight];
        const leftEyeBottom2 = [landmarks[373].x * imgWidth, landmarks[373].y * imgHeight];
        
        // Calculate absolute measurements
        const rightEyeWidth = this.calculateDistance(rightEyeInner, rightEyeOuter);
        const leftEyeWidth = this.calculateDistance(leftEyeInner, leftEyeOuter);
        
        const rightEyeHeight1 = this.calculateDistance(rightEyeTop, rightEyeBottom);
        const rightEyeHeight2 = this.calculateDistance(rightEyeTop2, rightEyeBottom2);
        const rightEyeHeight = (rightEyeHeight1 + rightEyeHeight2) / 2;
        
        const leftEyeHeight1 = this.calculateDistance(leftEyeTop, leftEyeBottom);
        const leftEyeHeight2 = this.calculateDistance(leftEyeTop2, leftEyeBottom2);
        const leftEyeHeight = (leftEyeHeight1 + leftEyeHeight2) / 2;
        
        // Inter-eye distance
        const interEyeDistance = this.calculateDistance(rightEyeInner, leftEyeInner);
        
        // Face width (for relative measurements)
        const faceLeft = [landmarks[234].x * imgWidth, landmarks[234].y * imgHeight];
        const faceRight = [landmarks[454].x * imgWidth, landmarks[454].y * imgHeight];
        const faceWidth = this.calculateDistance(faceLeft, faceRight);
        
        // Calculate RELATIVE measurements (scale-invariant)
        const avgEyeWidth = (rightEyeWidth + leftEyeWidth) / 2;
        
        measurements.right_eye_width_relative = rightEyeWidth / faceWidth;
        measurements.left_eye_width_relative = leftEyeWidth / faceWidth;
        measurements.avg_eye_width_relative = avgEyeWidth / faceWidth;
        
        // Eye Aspect Ratio (height/width) - naturally relative
        measurements.right_ear = rightEyeHeight / rightEyeWidth;
        measurements.left_ear = leftEyeHeight / leftEyeWidth;
        measurements.avg_ear = (measurements.right_ear + measurements.left_ear) / 2;
        
        // Inter-eye distance relative to eye width
        measurements.inter_eye_ratio = interEyeDistance / avgEyeWidth;
        
        // Calculate eye angles (upturned vs downturned)
        const rightAngle = Math.atan2(
            rightEyeOuter[1] - rightEyeInner[1],
            rightEyeOuter[0] - rightEyeInner[0]
        ) * (180 / Math.PI);
        
        const leftAngle = Math.atan2(
            leftEyeInner[1] - leftEyeOuter[1],
            leftEyeInner[0] - leftEyeOuter[0]
        ) * (180 / Math.PI);
        
        measurements.right_eye_angle = rightAngle;
        measurements.left_eye_angle = leftAngle;
        measurements.avg_eye_angle = (rightAngle + leftAngle) / 2;
        
        // Eyelid visibility (hooded detection)
        const rightCrease = [landmarks[157].x * imgWidth, landmarks[157].y * imgHeight];
        const leftCrease = [landmarks[384].x * imgWidth, landmarks[384].y * imgHeight];
        
        const rightLidVisibility = this.calculateDistance(rightCrease, rightEyeTop) / rightEyeHeight;
        const leftLidVisibility = this.calculateDistance(leftCrease, leftEyeTop) / leftEyeHeight;
        
        measurements.right_lid_visibility = rightLidVisibility;
        measurements.left_lid_visibility = leftLidVisibility;
        measurements.avg_lid_visibility = (rightLidVisibility + leftLidVisibility) / 2;
        
        // Symmetry score
        measurements.symmetry_score = 1 - Math.abs(rightEyeWidth - leftEyeWidth) / avgEyeWidth;
        
        return measurements;
    }
    
    classifyEyeShape(measurements) {
        const ear = measurements.avg_ear;
        const angle = measurements.avg_eye_angle;
        const lidVisibility = measurements.avg_lid_visibility;
        
        if (lidVisibility < 0.3) {
            return "Hooded";
        } else if (ear > 0.5) {
            return "Round";
        } else if (ear < 0.35) {
            if (angle > 2) return "Upturned";
            else if (angle < -2) return "Downturned";
            else return "Almond";
        } else {
            if (angle > 3) return "Upturned";
            else if (angle < -3) return "Downturned";
            else return "Almond";
        }
    }
    
    classifyEyeSize(measurements) {
        const eyeWidthRatio = measurements.avg_eye_width_relative;
        
        if (eyeWidthRatio < 0.15) return "Small";
        else if (eyeWidthRatio > 0.18) return "Large";
        else return "Medium";
    }
    
    classifyEyeSpacing(measurements) {
        const interEyeRatio = measurements.inter_eye_ratio;
        
        if (interEyeRatio < 0.95) return "Close-set";
        else if (interEyeRatio > 1.15) return "Wide-set";
        else return "Average-set";
    }
    
    calculateMatchScore(shape, size, spacing, productDetails) {
        let score = 100;
        
        // Perfect match for hooded eyes with specific products
        if (shape === "Hooded" && productDetails.suitable_shapes.includes("Hooded")) {
            score += 50;
        }
        
        // Prioritize universal products
        if (productDetails.suitable_sizes.length === 3) {
            score += 10;
        }
        if (productDetails.suitable_shapes.length >= 4) {
            score += 10;
        }
        
        // Cat eye styles work best for elongation
        if (["Round", "Downturned"].includes(shape) && productDetails.style_type.includes("Cat Eye")) {
            score += 20;
        }
        
        // Doll eye styles work best for round enhancement
        if (["Almond", "Upturned"].includes(shape) && productDetails.style_type.includes("Doll Eye")) {
            score += 15;
        }
        
        // Natural styles for subtle looks
        if (size === "Small" && productDetails.style_type.includes("Natural")) {
            score += 15;
        }
        
        return score;
    }
    
    recommendEyelashes(shape, size, spacing) {
        const recommendedProducts = [];
        
        for (const [category, products] of Object.entries(this.inventory)) {
            for (const [productName, details] of Object.entries(products)) {
                const sizeMatch = details.suitable_sizes.includes(size);
                const shapeMatch = details.suitable_shapes.includes(shape);
                
                if (sizeMatch && shapeMatch) {
                    recommendedProducts.push({
                        name: productName,
                        category: category,
                        style_type: details.style_type,
                        description: details.description,
                        intensity: details.intensity,
                        look: details.look,
                        match_score: this.calculateMatchScore(shape, size, spacing, details)
                    });
                }
            }
        }
        
        recommendedProducts.sort((a, b) => b.match_score - a.match_score);
        const topRecommendations = recommendedProducts.slice(0, 3);
        
        const spacingTips = {
            "Close-set": "Focus application on outer 2/3 of lash line to create width",
            "Wide-set": "Focus application on inner 2/3 of lash line to bring eyes closer",
            "Average-set": "Apply evenly across entire lash line for balanced look"
        };
        
        const shapeTips = {
            "Hooded": "Your hooded eyes look best with curled, wispy lashes that lift and open the eye. Avoid heavy styles.",
            "Round": "Elongate your beautiful round eyes with cat-eye styles that emphasize the outer corners.",
            "Almond": "Lucky you! Your almond eyes are versatile and can rock any lash style - go bold!",
            "Downturned": "Lift your eye shape with curled lashes that have extra volume at the outer corners.",
            "Upturned": "Balance your naturally lifted eyes with even length across the lash line."
        };
        
        return {
            top_picks: topRecommendations,
            all_suitable: recommendedProducts,
            application_tip: spacingTips[spacing] || "",
            shape_tip: shapeTips[shape] || "",
            total_matches: recommendedProducts.length
        };
    }
}

// Eyelash Try-On System
class EyelashTryOnSystem {
    constructor() {
        this.LEFT_EYE_UPPER = [246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7];
        this.RIGHT_EYE_UPPER = [466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249];
        this.LEFT_EYE_INNER = 133;
        this.LEFT_EYE_OUTER = 33;
        this.RIGHT_EYE_INNER = 362;
        this.RIGHT_EYE_OUTER = 263;
        
        this.faceMesh = null;
        this.initialized = false;
        this.eyelashImages = {};
    }
    
    async initialize() {
        if (this.initialized) return;
        
        this.faceMesh = new FaceMesh({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
            }
        });
        
        this.faceMesh.setOptions({
            maxNumFaces: 1,
            refineLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
        
        // Preload eyelash images
        await this.preloadEyelashImages();
        this.initialized = true;
    }
    
    async preloadEyelashImages() {
        const eyelashNames = [
            "Drunk In Love", "Wedding Day", "Foxy", "Flare", "Vixen",
            "Other Half 1", "Other Half 2", "Staycation", "Iconic"
        ];
        
        for (const name of eyelashNames) {
            try {
                const img = await this.loadImage(`eyelashes/${name}.png`);
                this.eyelashImages[name] = img;
            } catch (error) {
                console.warn(`Could not load eyelash image for ${name}:`, error);
            }
        }
    }
    
    loadImage(src) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = "anonymous";
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = src;
        });
    }
    
    rotateImage(canvas, angle) {
        if (angle === 0) return canvas;
        
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        
        const radians = angle * Math.PI / 180;
        const sin = Math.abs(Math.sin(radians));
        const cos = Math.abs(Math.cos(radians));
        
        const width = canvas.width;
        const height = canvas.height;
        
        const newWidth = width * cos + height * sin;
        const newHeight = width * sin + height * cos;
        
        tempCanvas.width = newWidth;
        tempCanvas.height = newHeight;
        
        tempCtx.translate(newWidth / 2, newHeight / 2);
        tempCtx.rotate(radians);
        tempCtx.drawImage(canvas, -width / 2, -height / 2);
        
        return tempCanvas;
    }
    
    getEyeRegionInfo(landmarks, eyeUpperIndices, innerIdx, outerIdx) {
        const upperPoints = eyeUpperIndices.map(i => landmarks[i]);
        const inner = landmarks[innerIdx];
        const outer = landmarks[outerIdx];
        
        const eyeWidth = Math.sqrt((outer.x - inner.x)**2 + (outer.y - inner.y)**2);
        const centerX = (inner.x + outer.x) / 2;
        const centerY = upperPoints.reduce((sum, point) => sum + point.y, 0) / upperPoints.length;
        
        return {
            centerX: centerX,
            centerY: centerY,
            width: eyeWidth
        };
    }
    
    async processEyelashTryOn(image, eyelashName, adjustments = {}) {
        if (!this.initialized) {
            await this.initialize();
        }
        
        const {
            verticalOffset = -10,
            horizontalOffset = 0,
            sizeScale = 2.0,
            heightScale = 1.0,
            rotationAngle = 0
        } = adjustments;
        
        const eyelashImg = this.eyelashImages[eyelashName];
        if (!eyelashImg) {
            throw new Error(`Eyelash image for ${eyelashName} not found`);
        }
        
        // Create canvas for the image
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = image.width;
        canvas.height = image.height;
        ctx.drawImage(image, 0, 0);
        
        // Convert to format for FaceMesh
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = image.width;
        tempCanvas.height = image.height;
        tempCtx.drawImage(image, 0, 0);
        
        const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        
        return new Promise((resolve, reject) => {
            this.faceMesh.onResults((results) => {
                if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
                    reject(new Error("No face detected in image"));
                    return;
                }
                
                const landmarks = results.multiFaceLandmarks[0];
                const width = canvas.width;
                const height = canvas.height;
                
                // Convert landmarks to pixel coordinates
                const pixelLandmarks = landmarks.map(lm => ({
                    x: lm.x * width,
                    y: lm.y * height
                }));
                
                const leftInfo = this.getEyeRegionInfo(
                    pixelLandmarks, 
                    this.LEFT_EYE_UPPER, 
                    this.LEFT_EYE_INNER, 
                    this.LEFT_EYE_OUTER
                );
                
                const rightInfo = this.getEyeRegionInfo(
                    pixelLandmarks,
                    this.RIGHT_EYE_UPPER,
                    this.RIGHT_EYE_INNER,
                    this.RIGHT_EYE_OUTER
                );
                
                const lashAspect = eyelashImg.width / eyelashImg.height;
                
                // Apply adjustments for left eye
                const lw = leftInfo.width * sizeScale;
                const lh = (lw / lashAspect) * heightScale;
                const lx = leftInfo.centerX - lw / 2 + horizontalOffset;
                const ly = leftInfo.centerY + verticalOffset - lh / 2;
                
                // Apply adjustments for right eye
                const rw = rightInfo.width * sizeScale;
                const rh = (rw / lashAspect) * heightScale;
                const rx = rightInfo.centerX - rw / 2 - horizontalOffset;
                const ry = rightInfo.centerY + verticalOffset - rh / 2;
                
                // Apply eyelashes with rotation
                this.applyEyelash(canvas, eyelashImg, rx, ry, rw, rh, -rotationAngle);
                
                // Flip for left eye and apply
                const flippedCanvas = document.createElement('canvas');
                const flippedCtx = flippedCanvas.getContext('2d');
                flippedCanvas.width = eyelashImg.width;
                flippedCanvas.height = eyelashImg.height;
                flippedCtx.scale(-1, 1);
                flippedCtx.drawImage(eyelashImg, -eyelashImg.width, 0);
                
                this.applyEyelash(canvas, flippedCanvas, lx, ly, lw, lh, rotationAngle);
                
                resolve(canvas);
            });
            
            this.faceMesh.send({image: imageData});
        });
    }
    
    applyEyelash(canvas, eyelashImg, x, y, width, height, rotationAngle) {
        const ctx = canvas.getContext('2d');
        
        // Create temporary canvas for rotation
        let rotatedEyelash = document.createElement('canvas');
        rotatedEyelash.width = eyelashImg.width;
        rotatedEyelash.height = eyelashImg.height;
        const tempCtx = rotatedEyelash.getContext('2d');
        tempCtx.drawImage(eyelashImg, 0, 0);
        
        // Apply rotation if needed
        if (rotationAngle !== 0) {
            rotatedEyelash = this.rotateImage(rotatedEyelash, rotationAngle);
        }
        
        // Draw the eyelash
        ctx.drawImage(rotatedEyelash, x, y, width, height);
    }
}

// Global instances
const recommender = new EyelashRecommendationSystem();
const tryOnSystem = new EyelashTryOnSystem();

// DOM elements
const imageInput = document.getElementById('imageInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const tryOnBtn = document.getElementById('tryOnBtn');
const resultCanvas = document.getElementById('resultCanvas');
const recommendationsDiv = document.getElementById('recommendations');
const loadingDiv = document.getElementById('loading');

// Event listeners for range inputs
document.getElementById('verticalOffset').addEventListener('input', updateValueDisplay);
document.getElementById('horizontalOffset').addEventListener('input', updateValueDisplay);
document.getElementById('sizeScale').addEventListener('input', updateValueDisplay);
document.getElementById('heightScale').addEventListener('input', updateValueDisplay);
document.getElementById('rotationAngle').addEventListener('input', updateValueDisplay);

function updateValueDisplay(e) {
    const target = e.target;
    const valueDisplay = document.getElementById(target.id + 'Value');
    valueDisplay.textContent = target.value;
}

let currentImage = null;
let analysisResults = null;

// Initialize the system
tryOnSystem.initialize().catch(console.error);

async function analyzeImage() {
    if (!imageInput.files[0]) {
        alert('Please select an image first');
        return;
    }
    
    showLoading(true);
    
    try {
        const image = await loadImageFromFile(imageInput.files[0]);
        currentImage = image;
        
        // Display original image
        const ctx = resultCanvas.getContext('2d');
        resultCanvas.width = image.width;
        resultCanvas.height = image.height;
        ctx.drawImage(image, 0, 0);
        
        // Analyze with FaceMesh for recommendations
        const analysis = await analyzeWithFaceMesh(image);
        analysisResults = analysis;
        
        displayRecommendations(analysis);
        tryOnBtn.disabled = false;
        
    } catch (error) {
        alert('Error analyzing image: ' + error.message);
        console.error(error);
    } finally {
        showLoading(false);
    }
}

async function tryOnEyelashes() {
    if (!currentImage) {
        alert('Please analyze an image first');
        return;
    }
    
    showLoading(true);
    
    try {
        const eyelashName = document.getElementById('eyelashSelect').value;
        const adjustments = {
            verticalOffset: parseInt(document.getElementById('verticalOffset').value),
            horizontalOffset: parseInt(document.getElementById('horizontalOffset').value),
            sizeScale: parseFloat(document.getElementById('sizeScale').value),
            heightScale: parseFloat(document.getElementById('heightScale').value),
            rotationAngle: parseFloat(document.getElementById('rotationAngle').value)
        };
        
        const resultCanvas = await tryOnSystem.processEyelashTryOn(
            currentImage, 
            eyelashName, 
            adjustments
        );
        
        const ctx = document.getElementById('resultCanvas').getContext('2d');
        document.getElementById('resultCanvas').width = resultCanvas.width;
        document.getElementById('resultCanvas').height = resultCanvas.height;
        ctx.drawImage(resultCanvas, 0, 0);
        
    } catch (error) {
        alert('Error applying eyelashes: ' + error.message);
        console.error(error);
    } finally {
        showLoading(false);
    }
}

function loadImageFromFile(file) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = URL.createObjectURL(file);
    });
}

function analyzeWithFaceMesh(image) {
    return new Promise((resolve, reject) => {
        const faceMesh = new FaceMesh({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
            }
        });
        
        faceMesh.setOptions({
            staticImageMode: true,
            maxNumFaces: 1,
            refineLandmarks: true,
            minDetectionConfidence: 0.5
        });
        
        faceMesh.onResults((results) => {
            if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
                reject(new Error("No face detected in image"));
                return;
            }
            
            const landmarks = results.multiFaceLandmarks[0];
            const measurements = recommender.getEyeMeasurements(landmarks, image.width, image.height);
            
            const eyeShape = recommender.classifyEyeShape(measurements);
            const eyeSize = recommender.classifyEyeSize(measurements);
            const eyeSpacing = recommender.classifyEyeSpacing(measurements);
            
            const recommendations = recommender.recommendEyelashes(eyeShape, eyeSize, eyeSpacing);
            
            resolve({
                classification: {
                    eye_shape: eyeShape,
                    eye_size: eyeSize,
                    eye_spacing: eyeSpacing
                },
                measurements: {
                    eye_aspect_ratio: Math.round(measurements.avg_ear * 1000) / 1000,
                    eye_angle: Math.round(measurements.avg_eye_angle * 100) / 100,
                    lid_visibility: Math.round(measurements.avg_lid_visibility * 1000) / 1000,
                    inter_eye_ratio: Math.round(measurements.inter_eye_ratio * 1000) / 1000,
                    eye_width_to_face_ratio: Math.round(measurements.avg_eye_width_relative * 1000) / 1000,
                    symmetry_score: Math.round(measurements.symmetry_score * 1000) / 1000
                },
                recommendations: recommendations
            });
        });
        
        // Convert image to format for FaceMesh
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = image.width;
        canvas.height = image.height;
        ctx.drawImage(image, 0, 0);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        
        faceMesh.send({image: imageData});
    });
}

function displayRecommendations(analysis) {
    const { classification, measurements, recommendations } = analysis;
    
    let html = `
        <h3>Eye Analysis Results</h3>
        <div class="recommendation">
            <p><strong>Shape:</strong> ${classification.eye_shape}</p>
            <p><strong>Size:</strong> ${classification.eye_size}</p>
            <p><strong>Spacing:</strong> ${classification.eye_spacing}</p>
            <p><strong>Symmetry Score:</strong> ${measurements.symmetry_score}</p>
        </div>
        
        <h3>Top Recommendations</h3>
    `;
    
    recommendations.top_picks.forEach((product, index) => {
        const isTopPick = index === 0;
        html += `
            <div class="recommendation ${isTopPick ? 'top-pick' : ''}">
                <h4>${product.name} (Match: ${product.match_score}%)</h4>
                <p><strong>Category:</strong> ${product.category}</p>
                <p><strong>Style:</strong> ${product.style_type}</p>
                <p><strong>Intensity:</strong> ${product.intensity}</p>
                <p><strong>Look:</strong> ${product.look}</p>
                <p>${product.description}</p>
                ${isTopPick ? '<p><strong>🌟 BEST MATCH</strong></p>' : ''}
            </div>
        `;
    });
    
    html += `
        <div class="recommendation">
            <h4>Application Tips</h4>
            <p><strong>Shape-specific:</strong> ${recommendations.shape_tip}</p>
            <p><strong>Spacing:</strong> ${recommendations.application_tip}</p>
        </div>
        
        <p><em>Found ${recommendations.total_matches} suitable eyelash styles</em></p>
    `;
    
    recommendationsDiv.innerHTML = html;
}

function showLoading(show) {
    loadingDiv.style.display = show ? 'block' : 'none';
}