import React, { useState, useEffect } from "react";
import axios from "axios";

const ModelsPage = () => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch models from backend
    axios
      .get("http://localhost:5000/api/models") // Adjust the endpoint based on your backend
      .then((response) => {
        setModels(response.data);
        setLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching models:", error);
        setLoading(false);
      });
  }, []);

  if (loading) return <p>Loading models...</p>;

  return (
    <div>
      <h1>Available Models</h1>
      <div className="models-list">
        {models.map((model, index) => (
          <div key={index} className="model-card">
            <h2>{model.name}</h2>
            <p>{model.description}</p>
            {model.isFunctional ? (
              <button onClick={() => alert(`Running ${model.name}...`)}>
                Run Model
              </button>
            ) : (
              <p>Coming Soon</p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default ModelsPage;
