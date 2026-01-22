🛒 Amazon Hybrid Recommender System
An End-to-End Machine Learning project that recommends electronic products to users based on their purchase history. It utilizes a Hybrid Architecture combining Collaborative Filtering (ALS) for candidate generation and Gradient Boosting (LightGBM) for re-ranking.

🚀 Live Demo
Frontend Dashboard (Streamlit): Click Here to Try the App

Backend API (Swagger UI): View API Docs

Note: The services are hosted on free-tier instances. Please allow 30-50 seconds for the server to wake up on the first request.

🏗️ Architecture
The system follows a classic Two-Stage Recommender pipeline to handle scalability:

Code snippet

graph LR
    A[User ID] --> B(Stage 1: Retrieval)
    B -->|Generates Top 50 Candidates| C(Stage 2: Ranking)
    C -->|Re-ranks Top 10| D[Final Recommendations]
    
    subgraph "Retrieval (ALS)"
    B -- Collaborative Filtering --> B1[Matrix Factorization]
    end
    
    subgraph "Ranking (LightGBM)"
    C -- Feature Engineering --> C1[User-Item Interaction Features]
    C1 --> C2[Learning to Rank]
    end
Retrieval Layer (ALS): Uses Alternating Least Squares (Implicit Library) to quickly fetch top 50 relevant items from thousands of products based on latent user factors.

Ranking Layer (LightGBM): A Gradient Boosting model re-ranks these 50 candidates by predicting the probability of interaction, using engineered features (e.g., dot product scores).

📊 Dataset & Performance
Dataset: Amazon Electronics Reviews

Metric: Recall@30 ≈ 5%

Challenge: High sparsity (many users have very few interactions).

🛠️ Tech Stack
Language: Python 3.10

ML Libraries: implicit (ALS), lightgbm (Ranking), pandas, scipy, scikit-learn

API: FastAPI, Pydantic, Uvicorn

Frontend: Streamlit

DevOps: Docker, Git LFS (Large File Storage), Render (Cloud Hosting)

📂 Project Structure
Bash

├── .github/              # GitHub Actions workflows
├── configs/              # Configuration files (data paths, hyperparams)
├── data/                 # Raw and processed data (ignored in Git)
├── models/               # Trained models (.pkl, .npz - tracked via Git LFS)
├── src/
│   ├── api/              # FastAPI application code
│   ├── preprocessing/    # Data cleaning and transformation pipelines
│   ├── retrieval/        # ALS model training scripts
│   ├── ranking/          # LightGBM training scripts
│   └── utils/            # Helper functions
├── Dockerfile            # Docker configuration for Backend
├── dashboard.py          # Streamlit Frontend application
├── requirements.txt      # Dependencies for Streamlit (Frontend)
├── requirements-backend.txt # Dependencies for Render (Backend)
└── README.md             # Project Documentation
⚙️ Local Setup
1. Clone the Repository
Bash

git clone https://github.com/YOUR_USERNAME/recommendation-system-deployment.git
cd recommendation-system-deployment
2. Install Dependencies
It is recommended to use a virtual environment.

Bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-backend.txt
3. Run the Backend (API)
Bash

uvicorn src.api.app:app --reload
# API will be available at http://127.0.0.1:8000/docs
4. Run the Frontend (Dashboard)
Open a new terminal and run:

Bash

streamlit run dashboard.py
# UI will open at http://localhost:8501
🐳 Deployment (MLOps)
The project is containerized using Docker to ensure consistency across environments.

Docker Build & Run
Bash

docker build -t recsys-app .
docker run -p 8000:8000 recsys-app
CI/CD Pipeline
Model Versioning: Large model artifacts (>100MB) are tracked using Git LFS.

Cloud Hosting:

Render: Automatically builds the Docker image from the main branch and deploys the API.

Streamlit Cloud: Hosts the frontend dashboard, connected directly to the GitHub repository.
