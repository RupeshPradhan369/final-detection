import axios from 'axios';

const API_BASE = 'http://127.0.0.1:8000/api';

export const predictNews = async (text) => {
  const response = await axios.post(`${API_BASE}/predict/`, { text });
  return response.data;
};

export const healthCheck = async () => {
  const response = await axios.get(`${API_BASE}/health/`);
  return response.data;
};

export const explainNews = async (text) => {
  const response = await axios.post(`${API_BASE}/explain/`, { text });
  return response.data;
};