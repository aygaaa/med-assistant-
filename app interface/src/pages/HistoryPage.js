import React, { useState, useEffect } from 'react';
import './HomePage.css';

const HomePage = () => {
  const [text, setText] = useState('');
  const projectBrief = "Welcome to MED Assistant, your AI-powered medical assistant for diagnosing diseases, analyzing sentiments, and assisting healthcare professionals.";

  useEffect(() => {
    let i = 0;
    const interval = setInterval(() => {
      if (i < projectBrief.length) {
        setText((prev) => prev + projectBrief[i]);
        i++;
      } else {
        clearInterval(interval);
      }
    }, 50);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="home-container">
      <div className="project-brief">
        <h1>{text}</h1>
      </div>
      <div className="image-section">
        <div className="C:\Users\FreeComp\OneDrive\Pictures\Screenshots 1\Screenshot (22).png">Image 1</div>
        <div className="image-placeholder">Image 2</div>
        <div className="image-placeholder">Image 3</div>
      </div>
    </div>
  );
};

export default HomePage;
