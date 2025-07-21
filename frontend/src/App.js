import React, { useState, useRef } from "react";
import axios from "axios";
import "./styles.css";

// Define API URL
const API_URL = "http://localhost:8000";

function App() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState("");
    const [processedUrl, setProcessedUrl] = useState("");
    const [loading, setLoading] = useState(false);
    const [success, setSuccess] = useState(false);
    const [message, setMessage] = useState("");
    const [dragActive, setDragActive] = useState(false);
    const [outputPath, setOutputPath] = useState("");
    const [forgeryMessage, setForgeryMessage] = useState("");
    const inputRef = useRef(null);

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileChange({ target: { files: [e.dataTransfer.files[0]] } });
        }
    };

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('image/') && !file.type.startsWith('video/')) {
            setMessage("❌ Please upload an image or video file");
            return;
        }

        setSelectedFile(file);
        setPreviewUrl(URL.createObjectURL(file));
        setProcessedUrl("");
        setSuccess(false);
        setMessage("");
        setOutputPath("");
        setForgeryMessage("");
    };

    const handleUpload = async (endpoint) => {
        setMessage("");
        setProcessedUrl(null);
        setOutputPath("");
        setForgeryMessage("");
        setLoading(true);

        if (!selectedFile) {
            setMessage("❌ Please select a file first");
            setLoading(false);
            return;
        }

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            const response = await axios.post(
                `${API_URL}/${endpoint}/`,
                formData,
                {
                    responseType: endpoint === "detect-video" ? "json" : "blob",
                    headers: {
                        "Content-Type": "multipart/form-data",
                    },
                }
            );

            if (endpoint === "detect-video") {
                // Handle video response
                setOutputPath(response.data.message.split(": ")[1]);
                if (response.data.forgery_detection) {
                    setForgeryMessage(response.data.forgery_detection.message);
                }
                setMessage("✅ Video processed successfully!");
            } else {
                // Handle image response
                const blob = new Blob([response.data], { type: "image/jpeg" });
                const outputUrl = URL.createObjectURL(blob);
                setProcessedUrl(outputUrl);
                
                // Convert blob to text to read the logs
                const text = await blob.text();
                const lines = text.split('\n');
                
                // Find output path from logs
                const outputLine = lines.find(line => line.includes("Processed image saved to:"));
                if (outputLine) {
                    const outputFilePath = outputLine.split("Processed image saved to:")[1].trim();
                    setOutputPath(outputFilePath);
                }
                
                // Find forgery result from logs
                const forgeryLine = lines.find(line => line.includes("Forgery detection result:"));
                if (forgeryLine) {
                    const forgeryResult = forgeryLine.split("Forgery detection result:")[1].trim();
                    setForgeryMessage(forgeryResult);
                }
                
                setMessage(
                    <span style={{ whiteSpace: 'pre-line' }}>
                        ✅ Image processed successfully!
                        <br/>Processed Image saved to 
                        <br/>backend/backend/output
                        {outputPath && `\nProcessed image saved to: ${outputPath}`}
                        {forgeryMessage && `\n${forgeryMessage}`}
                    </span>
                );
            }
            setSuccess(true);
        } catch (error) {
            console.error("Error:", error);
            setMessage(
                error.response?.data?.detail ||
                    "❌ Error processing file. Please try again."
            );
        } finally {
            setLoading(false);
        }
    };

    const handleButtonClick = () => {
        inputRef.current?.click();
    };

    return (
        <div className="container">
            <h1>SecureVision</h1>
            <p>Real-Time Weapon and Forgery Detection System</p>

            <div 
                className="upload-section"
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
            >
                <input
                    ref={inputRef}
                    type="file"
                    onChange={handleFileChange}
                    accept="image/*,video/*"
                    style={{ display: 'none' }}
                />
                
                <div 
                    className={`file-upload-label ${dragActive ? 'drag-active' : ''}`}
                    onClick={handleButtonClick}
                >
                    📁 Choose File or Drag & Drop
                </div>

                {selectedFile && (
                    <div className="file-info">
                        Selected: {selectedFile.name}
                    </div>
                )}

                {/* Preview Image or Video */}
                {previewUrl && selectedFile?.type.startsWith("image") && (
                    <img src={previewUrl} alt="Uploaded Preview" className="preview" />
                )}

                {previewUrl && selectedFile?.type.startsWith("video") && (
                    <video src={previewUrl} controls className="preview" />
                )}

                {/* Loader and Success Animation */}
                {loading && <div className="loader"></div>}
                {success && <div className="success-animation">✔</div>}

                <div className="button-group">
                    <button 
                        onClick={() => handleUpload("detect-image")}
                        disabled={loading || !selectedFile || !selectedFile.type.startsWith("image")}
                    >
                        🖼️ Process Image
                    </button>
                    <button 
                        onClick={() => handleUpload("detect-video")}
                        disabled={loading || !selectedFile || !selectedFile.type.startsWith("video")}
                    >
                        🎥 Process Video
                    </button>
                </div>
            </div>

            {/* Message after processing */}
            {message && <p className="success-message" style={{ whiteSpace: 'pre-line' }}>{message}</p>}

            {/* Processed Output */}
            {(processedUrl || outputPath) && (
                <div className="result-container">
                    {processedUrl && selectedFile?.type.startsWith("image") && (
                        <>
                            <img src={processedUrl} alt="Processed" className="output" />
                        </>
                    )}
                    
                    {outputPath && selectedFile?.type.startsWith("video") && (
                        <>
                            <div className="video-result">
                                <div className="output-path">
                                    ✅ Video processed and saved to: {outputPath}
                                </div>
                            </div>
                            {forgeryMessage && (
                                <div className={`forgery-result ${forgeryMessage.toLowerCase().includes('authentic') ? 'authentic' : 'forged'}`}>
                                    <p>Video is {forgeryMessage.toLowerCase().includes('authentic') ? 'authentic' : 'forged'} with {forgeryMessage.match(/\d+\.\d+/)?.[0]}% confidence</p>
                                </div>
                            )}
                        </>
                    )}
                </div>
            )}
        </div>
    );
}

export default App;
