import React, { useState, useEffect } from 'react';
import './App.css';
import BackgroundComponent from './components/BackgroundComponent';

function App() {
  const [isDarkMode, setIsDarkMode] = useState(true);

  const toggleBackground = () => {
    setIsDarkMode(!isDarkMode);
    document.body.className = isDarkMode ? 'light-mode' : 'dark-mode';
  };

  // Automatically stop webcam detection when switching services
  return (
    <div>
      <BackgroundComponent
        toggleBackground={toggleBackground}
        isDarkMode={isDarkMode}
      />
    </div>
  );
}

export default App;