📰 Fake News Detection System
A web application that detects whether a news article is real or fake using a fine-tuned XLM-RoBERTa NLP model with LIME explainability — so users don't just get a label, they get a reason.
> Group project | My role: **Backend Developer** (Django + DRF)
---
🧠 How It Works
User submits a news article or headline via the React frontend
The Django backend receives it through a REST API endpoint
The XLM-RoBERTa model classifies the content as Real or Fake
LIME generates a human-readable explanation of the prediction
Results are returned to the frontend with confidence scores
---
⚙️ Tech Stack
Layer	Technology
Backend	Python, Django, Django REST Framework
Frontend	React.js
ML Model	XLM-RoBERTa (multilingual transformer)
Explainability	LIME (Local Interpretable Model-Agnostic Explanations)
Database	SQLite
---
🚀 Getting Started
Prerequisites
Python 3.9+
Node.js 16+
pip
Backend Setup
```bash
# Clone the repo
git clone https://github.com/RupeshPradhan369/fake-news-detection.git
cd fake-news-detection/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Start server
python manage.py runserver
```
Frontend Setup
```bash
cd ../frontend
npm install
npm start
```
---
📡 API Endpoints
Method	Endpoint	Description
POST	`/api/predict/`	Submit article text for classification
GET	`/api/history/`	Get past predictions
Example Request
```json
POST /api/predict/
{
  "text": "Scientists discover cure for all diseases overnight"
}
```
Example Response
```json
{
  "prediction": "FAKE",
  "confidence": 0.94,
  "explanation": ["cure", "overnight", "all diseases"]
}
```
---
👥 Team
Name	Role
Rupesh Pradhan	Backend Developer (Django + DRF + ML integration)
Pawan Gurung	Frontend Developer (React.js)
Bhesh Bahadur Saru	Documentation & Testing
---
📄 License
This project was built as an academic group project at Mount Annapurna Campus, Pokhara.
