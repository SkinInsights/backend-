# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from keras.models import load_model
from keras.layers import Input


app = FastAPI()

# Specify allowed origins (replace with your frontend URL)
origins = [
    "https://skininsights.netlify.app",  # React development server
    # Add other allowed origins, such as your production frontend URL
]

# Add CORS middleware with the specified configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define the size of the input images
img_size = (224, 224)

# Load the trained model
model = load_model("skin_cancer_detection_model.h5")

# Ensure the model is compiled before use
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Define a root endpoint to confirm the server is running
@app.get("/")
async def root():
    return {"message": "Server is running"}
# Define a function to preprocess the input image
def preprocess_image(img):
    img = cv2.resize(img, img_size)  # Resize to expected dimensions
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize
    return img

# Define API endpoint for prediction
@app.post("/predict/")
async def predict_skin_cancer(file: UploadFile = File(...)):
    # Read the image from the uploaded file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Preprocess the image
    img = preprocess_image(img)
    
    # Make a prediction
    pred = model.predict(img)
    pred_label = "Cancer" if pred[0][0] > 0.5 else "Not Cancer"
    pred_prob = float(pred[0][0])
    
    # Prepare response
    response_body = {
        "prediction": pred_label,
        "probability_of_skin_cancer": pred_prob
    }
    
    return JSONResponse(content=response_body)

# Ensure the FastAPI app runs when the script is executed
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

