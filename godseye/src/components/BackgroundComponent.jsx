import React, { useRef, useState, useCallback, useEffect } from 'react';
import './BackgroundComponent.css';
import lightModeIcon from "../images/mode-light-icon.png";
import darkModeIcon from "../images/mode-dark-icon.png";
import lightFile from "../images/file-light-icon.png";
import darkFile from "../images/file-dark-icon.png";
import lightRTMP from "../images/rtmp-light-icon.png";
import darkRTMP from "../images/rtmp-dark-icon.png";
import Webcam from 'react-webcam';

const BackgroundComponent = ({ isDarkMode, toggleBackground }) => {
  const webcamRef = useRef(null);
  const detectRef = useRef(null);
  const [detectSrc, setDetectSrc] = useState(null);

  const mediaRecorderRef = useRef(null);
  const fileInputRef = useRef(null);
  const [detectingWebcam, setDetectingWebcam] = useState(false);

  const [capturing, setCapturing] = useState(false);
  const [recordedChunks, setRecordedChunks] = useState([]);
  const [isWebcam, setIsWebcam] = useState(false);
  const [isRTMP, setIsRTMP] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);

  const backendUrl = "http://0.0.0.0:8000";
  const backendVideoUrl = "http://0.0.0.0:80";

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
      handleStartDetection();
      setDetectingWebcam(false);
    }
    else if (webcamRef.current) {
      console.log("webcamref is working");
      setDetectingWebcam(true);
    }
    if (!capturing) {
      handleStartCaptureClick();
    }
  }

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

      mediaRecorderRef.current.ondataavailable = handleDataAvailable;
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
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const json = await response.json();
      const annotatedFrame = json.annotated_chunk;
      const blob = new Blob([annotatedFrame], {type: 'image/jpeg'});
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

  useEffect(() => {
    const mediaRecorder = mediaRecorderRef.current;
    if (mediaRecorderRef.current) {
      console.log("media ref works");
    }
    if (detectingWebcam && mediaRecorder) {
      console.log("Use Effect Hook is working");

      mediaRecorder.addEventListener("dataavailable", handleDataAvailable);
    }
    return () => {
      if (mediaRecorder) {
        mediaRecorder.removeEventListener("dataavailable", handleDataAvailable);
      }
    };
  }, [detectingWebcam, handleDataAvailable]);

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
      {detectingWebcam && (
        <div className="webcam-display-container">
          <h3>Webcam Video Detections:</h3>
          <video ref={detectRef} src={detectSrc} />
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
          <img src={processedImage} alt="Processed" className="uploaded-image" />
        </div>
      )}
    </div>
  );  
};

export default BackgroundComponent;
