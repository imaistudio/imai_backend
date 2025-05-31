# AI Intent Router API

A FastAPI-based service that provides natural language intent recognition and task routing powered by Claude 3.7 Sonnet. The API can handle various image processing tasks including upscaling, reframing, and AI-powered image generation.

## Features

- Natural language intent recognition
- Image upscaling and enhancement
- Product design composition
- Black mirror effects
- Health check endpoint
- Claude AI integration

## Prerequisites

- Python 3.8+
- FAL AI API key
- Cloudinary credentials
- Claude API key

## Setup

1. **Clone the repository:**
   ```sh
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Create a `.env` file** in the root directory with your API credentials:
   ```env
   FAL_KEY=your_fal_ai_api_key
   CLOUDINARY_CLOUD_NAME=your_cloud_name
   CLOUDINARY_API_KEY=your_api_key
   CLOUDINARY_API_SECRET=your_api_secret
   CLAUDE_API_KEY=your_claude_api_key
   ```

## Running Locally

1. **Start the FastAPI server:**
   ```sh
   uvicorn intent_router_api:app --reload
   ```

2. **Access the API documentation:**
   - Open your browser and go to `http://localhost:8000/docs`
   - This will show you the interactive Swagger UI documentation

## Testing with Postman

1. **Import the API Collection:**
   - Open Postman
   - Import the `ALL_API_TESTING+.MD` file from the project root
   - This contains all the API endpoints and example requests

2. **Set up environment variables:**
   - Create a new environment in Postman
   - Add the following variables:
     - `base_url`: `http://localhost:8000` (for local testing)
     - `api_key`: Your Claude API key

3. **Test the endpoints:**
   - Health Check: `GET {{base_url}}/api/v1/health`
   - Process Intent: `POST {{base_url}}/api/v1/process-intent`
   - Black Mirror: `POST {{base_url}}/api/v1/black-mirror`
   - Compose Product: `POST {{base_url}}/api/v1/compose-product`

## Deploying to Vercel

1. **Install Vercel CLI:**
   ```sh
   npm install -g vercel
   ```

2. **Login to Vercel:**
   ```sh
   vercel login
   ```

3. **Deploy the project:**
   ```sh
   vercel
   ```

4. **Set environment variables in Vercel:**
   - Go to your project settings in Vercel dashboard
   - Add all the environment variables from your `.env` file

5. **Update Postman environment:**
   - Create a new environment for production
   - Set `base_url` to your Vercel deployment URL

## Project Structure

```
.
├── app/
│   ├── services/         # Core service implementations
│   └── config.py         # Configuration settings
├── src/
│   ├── workflow/         # Workflow orchestrators
│   ├── api/             # API clients
│   └── utils/           # Utility functions
├── main_images/         # Input images directory
├── output/             # Output directory for processed images
├── intent_router_api.py # Main FastAPI application
├── requirements.txt    # Project dependencies
└── README.md          # Documentation
```

## API Endpoints

- `POST /api/v1/process-intent`: Main endpoint for processing user intents
- `POST /api/v1/black-mirror`: Apply black mirror effects to images
- `POST /api/v1/compose-product`: Compose product designs from multiple images
- `GET /api/v1/health`: Health check endpoint
- `POST /api/v1/test-claude`: Test endpoint for Claude AI integration

## Error Handling

The API uses standard HTTP status codes:
- 200: Success
- 400: Bad Request
- 401: Unauthorized
- 404: Not Found
- 500: Internal Server Error

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license information here] 