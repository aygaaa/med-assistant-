import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import ModelSelector from "./components/ModelSelector";
import Navbar from "./components/Navbar/Navbar";
import HomePage from "./pages/HomePage";
import ModelsPage from "./pages/ModelsPage";
import HistoryPage from "./pages/HistoryPage";
import Chatbot from "./components/Chatbot";



const Pneumonia = () => <h2 style={{ color: "white" }}>Pneumonia Detection Model</h2>;
const Skin = () => <h2 style={{ color: "white" }}>Skin Disease Detection Model</h2>;
const Sentiment = () => <h2 style={{ color: "white" }}>Sentiment Analysis Model</h2>;
const Diabetes = () => <h2 style={{ color: "white" }}>Diabetes Diagnosis Model</h2>;

function App() {
  const [text, setText] = useState('');
  const projectBrief =
    "Welcome to MED Assistant, your AI-powered medical assistant for diagnosing diseases, analyzing sentiments, and assisting healthcare professionals.";

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
  }, [projectBrief]);

  return (
    <Router>
      {/* Wrapper div with scrollable content */}
      <div style={{ height: "100vh", overflowY: "scroll", backgroundColor: "#000" }}>
        <Navbar />
        <div style={{ paddingTop: "80px", textAlign: "center", color: "white" }}>
          {/* Typewriter Effect */}
          <p style={{ fontSize: "18px", fontWeight: "bold", marginBottom: "30px" }}>{text}</p>
        </div>
        <div style={{ padding: "20px", textAlign: "center" }}>
          <Routes>
            <Route path="/" element={<ModelSelector />} />
            <Route path="/pneumonia" element={<Pneumonia />} />
            <Route path="/skin" element={<Skin />} />
            <Route path="/sentiment" element={<Sentiment />} />
            <Route path="/diabetes" element={<Diabetes />} />
            <Route path="/" element={<HomePage />} />
            <Route path="/models" element={<ModelsPage />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/chatbot" element={<Chatbot />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
