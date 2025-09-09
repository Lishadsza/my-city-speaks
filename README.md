# My City Speaks

My City Speaks is a web application that can identify the language of a speaker based on a short audio clip. It can currently identify English, Kannada, Tulu, and Hindi.

## Project Structure

The project is divided into two main parts:

-   `frontend`: A React-based user interface that allows users to record and upload audio files.
-   `backend`: A Flask-based server that handles audio processing, feature extraction, and language prediction.

## Setup and Installation

### Backend

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    -   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    -   On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Flask server:**
    ```bash
    flask run
    ```
    The server will start on `http://localhost:5000`.

### Frontend

1.  **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```

2.  **Install the required dependencies:**
    ```bash
    npm install
    ```

3.  **Run the React development server:**
    ```bash
    npm run dev
    ```
    The application will be available at `http://localhost:5173`.

## API Endpoints

### `POST /predict`

This endpoint accepts a single audio file and returns the predicted language.

-   **Request:**
    -   Method: `POST`
    -   Body: `multipart/form-data` with a single field named `file` containing the audio file.
-   **Response:**
    -   **Success (200):**
        ```json
        {
          "language": "kannada",
          "note": "This language is commonly spoken in Karnataka. (Based on language, not location prediction.)"
        }
        ```
    -   **Error (400):**
        ```json
        {
          "error": "No file uploaded"
        }
        ```
    -   **Error (500):**
        ```json
        {
          "error": "Error during prediction: <error message>"
        }
        ```
