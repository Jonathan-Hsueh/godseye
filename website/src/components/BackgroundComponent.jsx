import React, { useRef, useState, useCallback, useEffect } from 'react';
import './BackgroundComponent.css';
import lightModeIcon from "../images/mode-light-icon.png";
import darkModeIcon from "../images/mode-dark-icon.png";
import lightFile from "../images/file-light-icon.png";
import darkFile from "../images/file-dark-icon.png";
import lightRTMP from "../images/rtmp-light-icon.png";
import darkRTMP from "../images/rtmp-dark-icon.png";
import Webcam from 'react-webcam';
import * as tf from '@tensorflow/tfjs'

const BackgroundComponent = ({ isDarkMode, toggleBackground }) => {
  const webcamRef = useRef(null);
  const detectRef = useRef(null);
  const imgRef = useRef(null);
  const fileInputRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const modelRef = useRef(null);

  // State management
  const [detectingWebcam, setDetectingWebcam] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [isWebcam, setIsWebcam] = useState(false);
  const [isRTMP, setIsRTMP] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [detectSrc, setDetectSrc] = useState(null);
  const [streamUrl, setStreamUrl] = useState("");
  const [modelLoaded, setModelLoaded] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [classNames] = useState(["gun"]); // Update with your classes

  useEffect(() => {
    const loadModel = async () => {
      try {
        const model = await tf.loadGraphModel('/models/model.json');
        modelRef.current = model;
        setModelLoaded(true);
        console.log("Model loaded successfully");
        // Warmup model
        const dummyInput = tf.zeros([1, 640, 640, 3]);
        await model.predict(dummyInput).data();
        tf.dispose(dummyInput);
      } catch (error) {
        console.error("Model loading failed:", error);
      }
    };
    loadModel();
  }, []);

  // very important my children
  const calculateIOU = (boxA, boxB) => {
    const x1 = Math.max(boxA[0] - boxA[2]/2, boxB[0] - boxB[2]/2);
    const y1 = Math.max(boxA[1] - boxA[3]/2, boxB[1] - boxB[3]/2);
    const x2 = Math.min(boxA[0] + boxA[2]/2, boxB[0] + boxB[2]/2);
    const y2 = Math.min(boxA[1] + boxA[3]/2, boxB[1] + boxB[3]/2);
    
    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = boxA[2] * boxA[3];
    const areaB = boxB[2] * boxB[3];
    
    return intersection / (areaA + areaB - intersection);
  };

  const applyNMS = (boxes, scores, iouThreshold = 0.5) => {
    let sortedIndices = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a]);
    const selected = [];
    
    while (sortedIndices.length > 0) {
      const current = sortedIndices.shift();
      selected.push(current);
      
      sortedIndices = sortedIndices.filter(idx => {
        return calculateIOU(boxes[current], boxes[idx]) <= iouThreshold;
      });
    }
    return selected;
  };

  // Unified detection handler
  const detectObjects = async (imageElement) => {
    if (!modelRef.current) throw new Error("Model not loaded");
  
    // Declare variables outside try block
    let imgTensor, resized, input, prediction;
    
    try {
      imgTensor = tf.browser.fromPixels(imageElement);
      resized = tf.image.resizeBilinear(imgTensor, [640, 640]);
      input = resized.expandDims(0).toFloat().div(255);
  
      prediction = await modelRef.current.execute(input);
  
      // Handle YOLOv8 output format
      if (prediction instanceof tf.Tensor) {
        const totalElements = prediction.size;
        const featuresPerPrediction = 84;
        const numPredictions = totalElements / featuresPerPrediction;
        
        if (numPredictions % 1 !== 0) {
          throw new Error(`Invalid tensor size ${totalElements} for 84 features per prediction`);
        }
        
        const reshaped = prediction.reshape([numPredictions, featuresPerPrediction]);
        const data = await reshaped.array();
        
        return {
          boxes: data.map(p => p.slice(0, 4)),
          scores: data.map(p => p[4]),
          classes: data.map(p => p.slice(5).indexOf(Math.max(...p.slice(5))))
        };
      }
      
      // Handle multi-output models
      const outputs = Array.isArray(prediction) ? prediction : prediction.outputs;
      return {
        boxes: await outputs[0].array(),
        scores: await outputs[1].array(),
        classes: await outputs[2].array()
      };
    } finally {
      const tensorsToDispose = [imgTensor, resized, input, prediction].filter(t => t !== undefined);
      tf.dispose(tensorsToDispose);
    }
  };

  // Image processing
  // In handleFileImage function, update the detection handling:
const handleFileImage = async () => {
  if (!selectedFile || !modelLoaded) return;

  const img = new Image();
  img.src = URL.createObjectURL(selectedFile);
  
  img.onload = async () => {
    try {
      setProcessing(true);
      const detections = await detectObjects(img);
      
      // Apply Non-Maximum Suppression
      
      const keepIndices = applyNMS(detections.boxes, detections.scores); // problematic line
      const filteredBoxes = keepIndices.map(i => detections.boxes[i]);
      const filteredScores = keepIndices.map(i => detections.scores[i]);
      const filteredClasses = keepIndices.map(i => detections.classes[i]);
      
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      
      // Draw filtered detections
      filteredBoxes.forEach((box, index) => {
        const [x, y, w, h] = box;
        const score = filteredScores[index];
        
        // Coordinate conversion
        const scaleX = canvas.width / 640;
        const scaleY = canvas.height / 640;
        const left = (x - w/2) * scaleX;
        const top = (y - h/2) * scaleY;
        const width = w * scaleX;
        const height = h * scaleY;

        // Draw elements
        ctx.strokeStyle = '#FF0000';
        ctx.lineWidth = 3;
        ctx.strokeRect(left, top, width, height);
        
        ctx.fillStyle = '#FF0000';
        ctx.fillText(`gun ${(score * 100).toFixed(1)}%`, left + 5, top + 20);
      });
      
      canvas.toBlob((blob) => {
        const processedUrl = URL.createObjectURL(blob);
        setProcessedImage(processedUrl);
      }, 'image/jpeg', 0.9);
      
    } catch (error) {
      console.error("Detection error:", error);
      alert(`Detection failed: ${error.message}`);
    } finally {
      setProcessing(false);
    }
  };
};

// In processVideoFrame function, update similarly:
const processVideoFrame = async () => {
  if (!webcamRef.current || !detectingWebcam || !modelLoaded) return;

  try {
    const video = webcamRef.current.video;
    const detections = await detectObjects(video);
    
    // Apply NMS
    const keepIndices = applyNMS(detections.boxes, detections.scores);
    const filteredBoxes = keepIndices.map(i => detections.boxes[i]);
    const filteredScores = keepIndices.map(i => detections.scores[i]);

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    // Draw filtered detections
    filteredBoxes.forEach((box, index) => {
      const [x, y, w, h] = box;
      const score = filteredScores[index];
      
      const left = (x - w/2) * canvas.width;
      const top = (y - h/2) * canvas.height;
      const width = w * canvas.width;
      const height = h * canvas.height;

      ctx.strokeStyle = '#FF0000';
      ctx.lineWidth = 3;
      ctx.strokeRect(left, top, width, height);
      
      ctx.fillStyle = '#FF0000';
      ctx.fillText(`gun ${(score * 100).toFixed(1)}%`, left + 5, top + 20);
    });

    const stream = canvas.captureStream(25);
    if (detectRef.current) {
      detectRef.current.srcObject = stream;
    }
    
    requestAnimationFrame(processVideoFrame);
  } catch (error) {
    console.error("Video processing error:", error);
    setDetectingWebcam(false);
  }
};
  // Webcam toggle

  const handleToggleRTMP = () => setIsRTMP(!isRTMP);

  const handleButtonClick = () => {
    fileInputRef.current.click();
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    setSelectedFile(file);
    setProcessedImage(null);
    setDetectingWebcam(false);
  };

  const startDetectionButton = () => {
    if (selectedFile) {
      handleFileImage();
      setDetectingWebcam(false);
    }
    else if (webcamRef.current) {
      console.log("webcamref is working");
      setDetectingWebcam(true);
    }
    if (!capturing) {
      handleStartCaptureClick();
    }
  };

  // Webcam/RTMP Processing
  ;

  // RTMP Stream Handling
  const handleRTMPStream = async () => {
    if (!streamUrl) return;
    
    const video = document.createElement('video');
    video.src = streamUrl;
    video.playsInline = true;
    
    video.onplay = async () => {
      const processFrame = async () => {
        if (video.paused || video.ended) return;
        
        const detections = await detectObjects(video);
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);

        detections.boxes.forEach((box, index) => {
          if (detections.scores[index] > 0.5) {
            const [x, y, w, h] = box;
            ctx.strokeStyle = '#FF0000';
            ctx.lineWidth = 2;
            ctx.strokeRect(
              x * canvas.width,
              y * canvas.height,
              w * canvas.width,
              h * canvas.height
            );
          }
        });

        setDetectSrc(canvas.toDataURL());
        requestAnimationFrame(processFrame);
      };
      requestAnimationFrame(processFrame);
    };
  };
  


  const handleStartDetection = async () => {
    if (!selectedFile) {
      alert("Please upload a file before starting detection.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(backendUrl + "/upload/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      console.log("Received Blob:", blob);
      setProcessedImage(imageUrl);
      console.log("Generated Image URL:", imageUrl);
    } catch (error) {
      console.error("Error processing file:", error);
    }
  };

  const handleStartCaptureClick = useCallback(() => {
    if (webcamRef.current && webcamRef.current.stream != null) {
      setCapturing(true);
      mediaRecorderRef.current = new MediaRecorder(webcamRef.current.stream, {
        mimeType: "video/webm"
      });

      mediaRecorderRef.current.start(1000);
    }
  }, [webcamRef]);

  const handleStopCaptureClick = useCallback(() => {
    setTimeout(() => {
      mediaRecorderRef.current.stop();
      setCapturing(false);
      setDetectingWebcam(false);
    }, 2000);
  }, []);
// mediarecorderref = handledataavailable so it gets the data chunk

  const handleDataAvailable = async (event) => {
    console.log("Before part is running");    
    if (event.data.size > 0) {
      console.log("This part is running");
      await sendChunkToServer(event.data);
    }
  };

  // webcam video chunk post request
  const sendChunkToServer = async (chunk) => {
    if (!detectingWebcam) {
      return;
    }
    if (!webcamRef.current) {
      alert("Please turn on webcam before detection.");
      return;
    }
    try {
      const formData = new FormData();
      formData.append("video_chunk", new Blob([chunk], {type: "video/webm"
    }));

      const response = await fetch(backendVideoUrl + "/upload_video_chunk", {
        method: 'POST', 
        body: formData,
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      // In sendChunkToServer()
      const json = await response.json();
      const annotatedFrame = json.annotated_chunk;

      // Convert base64 to Blob
      const byteCharacters = atob(annotatedFrame);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: 'image/jpeg' });

      // Update video source
      const url = URL.createObjectURL(blob);
      setDetectSrc(url);
    } catch (error) {
      console.error("Error during POST and blob conversion:", error);
      throw error;
    }
  }

  useEffect(() =>{
    if (detectSrc) {
      const videoElement = detectRef.current;
      videoElement.src = detectSrc;
      videoElement.play();
    }
  }, [detectSrc]);

  const handleToggleWebcam = () => {
    setSelectedFile(null);
    setProcessedImage(null);
    setIsWebcam(!isWebcam);
    setDetectingWebcam(false);
    if (!capturing) {
      handleStartCaptureClick();
    } else {
      handleStopCaptureClick();
    }
  };

  // Ensure the page extends when an image is uploaded
  useEffect(() => {
    if (selectedFile || processedImage) {
      document.documentElement.style.height = "auto";
      document.body.style.height = "auto";
    }
  }, [selectedFile, processedImage]);

  return (
    <div className={`background-wrapper ${isDarkMode ? 'dark-mode' : 'light-mode'}`}>
      <div className={`layered-header ${isDarkMode ? 'dark-header' : 'light-header'}`}>
        <p className="header-text">godseye.</p>
      </div>
      <div className={`layered-box ${isDarkMode ? 'dark-box' : 'light-box'}`}>
        <div className="inside-box">
          {isWebcam ? (
            <Webcam ref={webcamRef} audio={false} className="webcam-video" />
          ) : (
            <div className="icon-box">
              <button className="file-btn" onClick={handleButtonClick}>
                <img
                  src={isDarkMode ? lightFile : darkFile}
                  alt="File Upload Icon"
                  className="file-icon"
                />
                <input 
                  ref={fileInputRef}
                  type="file"
                  name="image"
                  onChange={handleFileChange}
                  style={{ display: 'none' }}
                />
              </button>
              <button className="rtmp-btn" onClick={handleToggleRTMP}>
                <img
                  src={isDarkMode ? lightRTMP : darkRTMP}
                  alt="RTMP Stream Icon"
                  className="rtmp-icon"
                />
              </button>
            </div>
          )}
        </div>
        <div className="selection-container">
          <button
            className={`webcam-button ${isDarkMode ? 'dark-webcam' : 'light-webcam'}`}
            onClick={handleToggleWebcam}
          >
            {isWebcam ? <span>Turn off Webcam</span> : <span>Turn on Webcam</span>}
          </button>
          <button
            className={`detection-button ${isDarkMode ? 'dark-dtbtn' : 'light-dtbtn'}`}
            onClick={startDetectionButton}
          >
            Start Detection
          </button>
        </div>
      </div>
      <div className="button-container">
        <button className="toggle-circle" onClick={toggleBackground}>
          <img
            src={isDarkMode ? lightModeIcon : darkModeIcon}
            alt="Mode Toggle Icon"
            className="toggle-icon"
          />
        </button>
      </div>
      {isRTMP && (
        <div className="rtmp-container">
          <input
            type="text"
            value={streamUrl}
            onChange={(e) => setStreamUrl(e.target.value)}
            placeholder="Enter RTMP stream URL"
          />
          <button onClick={handleRTMPStream}>Start RTMP Processing</button>
        </div>
      )}
      {detectingWebcam && (
        <div className="webcam-display-container">
          <h3>Webcam Video Detections:</h3>
          <video ref={detectRef} src={detectSrc} autoPlay muted/>
        </div>
      )}
      {selectedFile && (
        <div className="image-display-container">
          <h3>Uploaded Image Preview:</h3>
          <img src={URL.createObjectURL(selectedFile)} alt="Uploaded" className="uploaded-image" />
        </div>
      )}
      {processedImage && (
        <div className="image-display-container">
          <h3>Processed Image:</h3>
          <img 
            ref={imgRef} 
            src={processedImage} 
            alt="Processed" 
            className="uploaded-image"
          />
        </div>
      )}
    </div>
  );  
};

export default BackgroundComponent;
