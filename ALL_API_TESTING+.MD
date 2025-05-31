# AI Intent Router API – Documentation

## Base URL
```
http://localhost:8000
```

---

## 1. Process Intent

### Endpoint
```
POST /api/v1/process-intent
```

### Description
Main entry point for all user interactions. Accepts user input (prompt) and/or images, determines the intent, and routes to the appropriate workflow.

### Request

- **Content-Type:** `multipart/form-data`
- **Fields:**
  - `user_input` (string, optional): The user's prompt or instruction. If omitted and at least two images are provided, routes to product composition.
  - `context` (string, optional): JSON string for additional context.
  - `files` (file[], optional): One or more images (product, design, color, etc.).

#### Example 1: Product Compose (no prompt, just images)
```http
POST /api/v1/process-intent
Content-Type: multipart/form-data

files: [product.png, design.png, color.png]
```

#### Example 2: Intent with Prompt and Images
```http
POST /api/v1/process-intent
Content-Type: multipart/form-data

user_input: "Make this shoe black with gold accents"
files: [product.png, color.png]
```

### Success Response
```json
{
  "status": "success",
  "result": {
    "status": "success",
    "output_dir": "output/20250531_153802_394808",
    "original_prompt": "...",
    "enhanced_prompt": "...",
    "generated_images": ["output/20250531_153802_394808/edited_product_1.png"],
    "metadata_path": "output/20250531_153802_394808/metadata_20250531_153904_864031.json",
    "analysis": { ... }
  }
}
```

### Error Responses
- **400 Bad Request:** Missing required images or prompt.
- **500 Internal Server Error:** Unexpected error.

---

## 2. Compose Product

### Endpoint
```
POST /api/v1/compose-product
```

### Description
Compose a new product design using different combinations of images and an optional prompt.

### Request

- **Content-Type:** `multipart/form-data`
- **Fields:**
  - `product` (file, optional): Product image.
  - `design` (file, optional): Design inspiration image.
  - `color` (file, optional): Color palette image.
  - `prompt` (string, optional): Text prompt for the design.

#### Supported Scenarios
- **All three images:** Full composition.
- **Product + Color:** Keeps original design, applies color.
- **Product + Design:** Uses design for both color and design.
- **Color + Design:** Requires prompt.

#### Example: Full Composition
```http
POST /api/v1/compose-product
Content-Type: multipart/form-data

product: [product.png]
design: [design.png]
color: [color.png]
prompt: "Make this shoe look retro with orange and yellow swirls"
```

#### Example: Product + Color
```http
POST /api/v1/compose-product
Content-Type: multipart/form-data

product: [product.png]
color: [color.png]
```

### Success Response
```json
{
  "status": "success",
  "result": {
    "status": "success",
    "output_dir": "...",
    "generated_images": ["..."],
    "metadata_path": "...",
    "analysis": { ... }
  }
}
```

### Error Responses
- **400 Bad Request:** Not enough images, or missing prompt for color+design.
- **500 Internal Server Error:** Unexpected error.

---

## 3. Black Mirror

### Endpoint
```
POST /api/v1/black-mirror
```

### Description
Remix a single image in a creative way using a prompt.

### Request

- **Content-Type:** `multipart/form-data`
- **Fields:**
  - `image` (file, required): The image to remix.
  - `prompt` (string, optional): Remix instructions.

#### Example
```http
POST /api/v1/black-mirror
Content-Type: multipart/form-data

image: [shoe.png]
prompt: "Make this look like a cyberpunk artifact"
```

### Success Response
```json
{
  "status": "success",
  "result": {
    "status": "success",
    "input_image": "...",
    "enhanced_prompt": "...",
    "generated_image": "...",
    ...
  }
}
```

### Error Responses
- **400 Bad Request:** Image missing.
- **500 Internal Server Error:** Unexpected error.

---

## 4. Test Claude

### Endpoint
```
POST /api/v1/test-claude
```

### Description
Send a prompt to the Claude model and get a text response (for debugging or prompt engineering).

### Request

- **Content-Type:** `application/json`
- **Body:**
  - `prompt` (string, required): The prompt to send.

#### Example
```json
{
  "prompt": "Describe a futuristic sneaker"
}
```

### Success Response
```json
{
  "response": "A futuristic sneaker might feature..."
}
```

### Error Responses
- **500 Internal Server Error:** Unexpected error.

---

## 5. Health Check

### Endpoint
```
GET /api/v1/health
```

### Description
Check if the API is running.

### Success Response
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

---

## 6. (If present) Elemental Design Creation

### Endpoint
```
POST /api/v1/elemental-design-creation
```

### Description
Create a new design based on three input images and a user directive.

### Request

- **Content-Type:** `multipart/form-data`
- **Fields:**
  - `image1` (file, required)
  - `image2` (file, required)
  - `image3` (file, required)
  - `user_directive` (string, required)
  - `size` (string, optional, default: "1024x1024")
  - `quality` (string, optional, default: "standard")
  - `style` (string, optional)
  - `n_images` (int, optional, default: 1)
  - `output_format` (string, optional, default: "png")

#### Example
```http
POST /api/v1/elemental-design-creation
Content-Type: multipart/form-data

image1: [img1.png]
image2: [img2.png]
image3: [img3.png]
user_directive: "Combine these into a modern abstract pattern"
```

### Success Response
```json
{
  "status": "success",
  "output_dir": "...",
  "created_elemental_designs": ["..."],
  "metadata_path": "...",
  "user_directive": "...",
  "final_prompt": "..."
}
```

### Error Responses
- **400 Bad Request:** Missing or invalid parameters.
- **500 Internal Server Error:** Unexpected error.

---

## 7. (If present) Analyze and Generate

### Endpoint
```
POST /analyze-and-generate/
```

### Description
Analyze multiple images and generate a new design based on their elements.

### Request

- **Content-Type:** `multipart/form-data`
- **Fields:**
  - `images` (file[], required): List of images to analyze.

#### Example
```http
POST /analyze-and-generate/
Content-Type: multipart/form-data

images: [img1.png, img2.png, img3.png]
```

### Success Response
```json
{
  "status": "success",
  "design_elements": { ... },
  "concatenated_image": "...",
  "generated_image": "..."
}
```

### Error Responses
- **400/500:** As appropriate.

---

## Special Intents via `/api/v1/process-intent`

The following advanced workflows are accessible by sending the right prompt and files to `/api/v1/process-intent`. These do not have their own direct endpoints, but are triggered by intent analysis.

### 1. Kling (Video Generation)
- **Intent:** `kling`
- **How to trigger:**
  - **Prompt:** "Make a video from this image" or "Animate this image"
  - **Files:** 1 image file
- **Example Request:**
```http
POST /api/v1/process-intent
Content-Type: multipart/form-data

user_input: "Make a video from this image"
files: [shoe.png]
```
- **Response:**
```json
{
  "status": "success",
  "workflow_result": {
    "status": "success",
    "message": "Video generated successfully",
    "video_path": "...",
    "cloudinary_url": "..."
  }
}
```

---

### 2. Clarity (Clarity Upscaler)
- **Intent:** `edit_image` (with operation clarify/clarity)
- **How to trigger:**
  - **Prompt:** "Enhance the clarity of this image" or "clarity upscale"
  - **Files:** 1 image file
- **Example Request:**
```http
POST /api/v1/process-intent
Content-Type: multipart/form-data

user_input: "clarity upscale"
files: [shoe.png]
```
- **Response:**
```json
{
  "status": "success",
  "workflow_result": {
    "status": "success",
    "imageUrl": "...",
    "localPath": "..."
  }
}
```

---

### 3. AuraSR (Upscaler)
- **Intent:** `aurasr`
- **How to trigger:**
  - **Prompt:** "Upscale this image to 4K" or "super resolution"
  - **Files:** 1 image file
- **Example Request:**
```http
POST /api/v1/process-intent
Content-Type: multipart/form-data

user_input: "Upscale this image to 4K"
files: [shoe.png]
```
- **Response:**
```json
{
  "status": "success",
  "workflow_result": {
    "status": "success",
    "imageUrl": "...",
    "localPath": "..."
  }
}
```

---

### 4. Reframe
- **Intent:** `reframe`
- **How to trigger:**
  - **Prompt:** "Reframe this image to square" or "outpaint this image"
  - **Files:** 1 image file
- **Example Request:**
```http
POST /api/v1/process-intent
Content-Type: multipart/form-data

user_input: "Reframe this image to square"
files: [shoe.png]
```
- **Response:**
```json
{
  "status": "success",
  "workflow_result": {
    "status": "success",
    "original_image": "...",
    "reframed_image": "...",
    "reframed_url": "..."
  }
}
```

---

### 5. Analyzer
- **Intent:** `analyze_image`
- **How to trigger:**
  - **Prompt:** "Analyze this image"
  - **Files:** 1 image file
- **Example Request:**
```http
POST /api/v1/process-intent
Content-Type: multipart/form-data

user_input: "Analyze this image"
files: [shoe.png]
```
- **Response:**
```json
{
  "status": "success",
  "workflow_result": {
    "status": "success",
    "analysis": { ... }
  }
}
```

---

### 6. Comparer
- **Intent:** `compare_images`
- **How to trigger:**
  - **Prompt:** "Compare these images"
  - **Files:** 2 or 3 image files
- **Example Request:**
```http
POST /api/v1/process-intent
Content-Type: multipart/form-data

user_input: "Compare these images"
files: [shoe1.png, shoe2.png]
```
- **Response:**
```json
{
  "status": "success",
  "workflow_result": {
    "status": "success",
    "comparison": { ... }
  }
}
```

---

### 7. Elemental Design Creation
- **Intent:** `elemental_design_creation`
- **How to trigger:**
  - **Prompt:** "elemental design creation"
  - **Files:** 3 image files
- **Example Request:**
```http
POST /api/v1/process-intent
Content-Type: multipart/form-data

user_input: "elemental design creation"
files: [img1.png, img2.png, img3.png]
```
- **Response:**
```json
{
  "status": "success",
  "workflow_result": {
    "status": "success",
    "output_dir": "...",
    "created_elemental_designs": ["..."],
    "metadata_path": "...",
    "user_directive": "...",
    "final_prompt": "..."
  }
}
```

---

## Direct API Endpoints and Intent Router Accessibility

| Endpoint                        | Direct API | Accessible via Intent Router? |
|----------------------------------|:----------:|:----------------------------:|
| /api/v1/process-intent           |    Yes     |            Yes               |
| /api/v1/compose-product          |    Yes     |            Yes               |
| /api/v1/elemental-design-creation|    Yes     |            Yes               |
| /api/v1/black-mirror             |    Yes     |             No               |
| /api/v1/test-claude              |    Yes     |             No               |
| /api/v1/health                   |    Yes     |             No               |
| /analyze-and-generate/           |    Yes     |             No               |

---

**To use advanced features, use `/api/v1/process-intent` with the right prompt and files as shown above.**

## General Notes

- All endpoints return JSON responses.
- For image upload fields, use `multipart/form-data` and attach files as needed.
- Error responses will include a `detail` field with a human-readable message.
- For best results, use descriptive filenames for images (e.g., `product.png`, `color.png`, `design.png`).

---

**This documentation is suitable for importing into Postman or any API client. If you need OpenAPI/Swagger or Postman collection JSON, let me know!** 