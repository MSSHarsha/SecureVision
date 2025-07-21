from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
from ultralytics import YOLO
from fastapi.responses import FileResponse, Response
import tempfile
import torch
import logging
import shutil
from fastapi import HTTPException
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image, ImageChops, ImageEnhance
import tensorflow as tf
from tensorflow.keras.models import load_model
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# === Torch Debug Info ===
print("Torch CUDA Available:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# === Load YOLO model ===
logger.info("Loading YOLO model...")
model = YOLO("epoch90.pt")

# Configure model settings
if hasattr(model, 'predictor') and model.predictor is not None:
    # Set model configuration
    model.predictor.args.device = device
    model.predictor.args.conf = 0.25  # Confidence threshold
    model.predictor.args.iou = 0.45   # NMS IoU threshold
    model.predictor.args.agnostic_nms = True  # Use agnostic NMS
    model.predictor.args.max_det = 300  # Maximum number of detections
    model.predictor.args.classes = None  # Detect all classes
    model.predictor.args.retina_masks = False  # Disable retina masks
    model.predictor.args.verbose = False  # Disable verbose output
else:
    logger.warning("Model predictor not initialized properly")

# === Ensure output directory exists ===
os.makedirs("backend/output", exist_ok=True)

# === Health Check Endpoint ===
@app.get("/")
def read_root():
    return {"message": "SecureVision Backend is running"}

# === Helper Functions ===
def process_results(image, results):
    """Process YOLO detection results and draw bounding boxes on the image."""
    try:
        # Get the annotated image from the results
        annotated_image = results[0].plot()
        return annotated_image
    except Exception as e:
        logger.error(f"Error processing results: {str(e)}")
        # If there's an error, return the original image
        return image

# === Image Detection Endpoint ===
@app.post("/detect-image/")
async def detect_image(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_path = f"uploads/{file.filename}"
        os.makedirs("uploads", exist_ok=True)
        
        # Read file content
        content = await file.read()
        
        # Save the image directly without validation
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        # Load image with OpenCV
        image = cv2.imread(file_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to load image with OpenCV")

        # Run forgery detection
        try:
            # Create a new UploadFile object for forgery detection
            forgery_file = UploadFile(
                filename=file.filename,
                file=io.BytesIO(content)
            )
            forgery_result = await detect_forgery(forgery_file)
        except Exception as e:
            logger.error(f"Forgery detection failed: {str(e)}")
            # Provide a default result if forgery detection fails
            forgery_result = {
                "message": "Forgery detection unavailable",
                "is_forged": False,
                "confidence": 0.0
            }
        
        # Run weapon detection
        results = model(image)
        
        # Process results
        processed_image = process_results(image, results)
        
        # Save processed image
        output_path = f"output/processed_{file.filename}"
        os.makedirs("output", exist_ok=True)
        cv2.imwrite(output_path, processed_image)
        logger.info(f"Processed image saved to: {output_path}")
        
        # Read the processed image
        with open(output_path, "rb") as f:
            image_bytes = f.read()
        
        # Create response with forgery detection headers
        response = Response(content=image_bytes, media_type="image/jpeg")
        response.headers["x-forgery-result"] = forgery_result["message"]
        response.headers["x-forgery-is-forged"] = str(forgery_result["is_forged"]).lower()
        response.headers["x-forgery-confidence"] = str(forgery_result["confidence"])
        response.headers["x-output-path"] = os.path.abspath(output_path)
        
        return response
        
    except Exception as e:
        logger.error(f"Error in detect_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# === Load the forgery detection model ===
try:
    forgery_model = load_model('model.keras', compile=False)
    logger.info("Forgery detection model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load forgery detection model: {str(e)}")
    forgery_model = None

def difference(image_path):
    """Calculate the difference between original and resaved image."""
    try:
        # Create a temporary file for the resaved image
        temp_dir = tempfile.gettempdir()
        resaved_name = os.path.join(temp_dir, 'resaved.jpg')
        
        # Open and resave the image
        try:
            org = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to open image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to open image: {str(e)}")
            
        org.save(resaved_name, 'JPEG', quality=92)
        resaved = Image.open(resaved_name)
        
        # Calculate difference
        diff = ImageChops.difference(org, resaved)
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        
        # Enhance the difference
        diff = ImageEnhance.Brightness(diff).enhance(scale)
        
        # Clean up temporary file
        try:
            os.remove(resaved_name)
        except:
            pass
            
        return diff
    except Exception as e:
        logger.error(f"Error in difference calculation: {str(e)}")
        raise

@app.post("/detect-forgery/")
async def detect_forgery(file: UploadFile = File(...)):
    """Endpoint to detect image forgery."""
    if forgery_model is None:
        raise HTTPException(status_code=500, detail="Forgery detection model not loaded")
        
    try:
        logger.info(f"Processing image for forgery detection: {file.filename}")
        
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        # Read file content
        content = await file.read()
        
        # Save the image directly without validation
        image_path = f"uploads/{file.filename}"
        with open(image_path, "wb") as buffer:
            buffer.write(content)
        logger.info(f"Saved uploaded image to: {image_path}")
        
        # Process the image for forgery detection
        try:
            # Calculate difference
            diff_image = difference(image_path)
            
            # Resize and prepare for model
            img_array = np.array(diff_image.resize((128, 128))).flatten() / 255.0
            img_array = img_array.reshape(-1, 128, 128, 3)
            
            # Make prediction
            pred = forgery_model.predict(img_array)[0]
            
            # Determine result
            is_forged = bool(pred[1] > pred[0])  # Convert to Python bool
            confidence = float(pred[1] if is_forged else pred[0])  # Convert to Python float
            
            # Clean up
            try:
                os.remove(image_path)
            except:
                pass
                
            # Return result
            result = {
                "success": True,
                "is_forged": is_forged,
                "confidence": confidence,
                "message": f"Image is {'forged' if is_forged else 'not forged'} with {confidence:.2%} confidence"
            }
            
            logger.info(f"Forgery detection result: {result['message']}")
            return result
            
        except Exception as e:
            logger.error(f"Error in forgery detection: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in forgery detection: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# === Video Detection Endpoint ===
@app.post("/detect-video/")
async def detect_video(file: UploadFile = File(...)):
    try:
        logger.info(f"Processing video: {file.filename}")
        
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("output", exist_ok=True)
        
        # Save the uploaded video
        video_path = f"uploads/{file.filename}"
        with open(video_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        logger.info(f"Saved uploaded video to: {video_path}")

        # Process the video
        output_path = f"output/processed_{file.filename}"
        
        # Process video with YOLO
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Failed to open video file")
            raise HTTPException(status_code=500, detail="Failed to open video file")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Create VideoWriter with H.264 codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            logger.error("Failed to create output video file")
            raise HTTPException(status_code=500, detail="Failed to create output video file")

        frame_count = 0
        detection_count = 0
        forgery_result = None

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                
                # Run YOLO detection on frame
                results = model(frame)
                processed_frame = process_results(frame, results)
                
                # Count detections in this frame
                detection_count += len(results[0].boxes)
                
                # Write the processed frame
                out.write(processed_frame)
                
                # For the first frame, perform forgery detection
                if frame_count == 1:
                    # Save the frame as an image
                    sample_frame_path = f"uploads/sample_frame_{file.filename}.jpg"
                    cv2.imwrite(sample_frame_path, frame)
                    
                    # Perform forgery detection on the first frame
                    try:
                        with open(sample_frame_path, "rb") as img_file:
                            forgery_file = UploadFile(
                                filename=f"frame_{file.filename}.jpg",
                                file=img_file
                            )
                            forgery_result = await detect_forgery(forgery_file)
                    except Exception as e:
                        logger.error(f"Forgery detection failed: {str(e)}")
                        forgery_result = {
                            "message": "Forgery detection unavailable",
                            "is_forged": False,
                            "confidence": 0.0
                        }
                    
                    # Clean up the sample frame
                    try:
                        os.remove(sample_frame_path)
                        logger.info(f"Removed sample frame: {sample_frame_path}")
                    except:
                        pass
                
                # Log progress every 10% of frames
                if frame_count % max(1, total_frames // 10) == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")

        finally:
            cap.release()
            out.release()

        # Clean up the uploaded file
        try:
            os.remove(video_path)
            logger.info(f"Removed temporary file: {video_path}")
        except:
            pass

        logger.info(f"Video processing complete. Processed {frame_count} frames with {detection_count} total detections.")
        logger.info(f"Output saved to: {output_path}")

        # Return the results
        return {
            "success": True,
            "message": f"Output saved to: {output_path}",
            "weapon_detections": detection_count,
            "frames_processed": frame_count,
            "forgery_detection": forgery_result
        }

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
