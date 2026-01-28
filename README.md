# Amazon Hybrid Recommender System
An End-to-End Machine Learning project that recommends electronic products to users based on their purchase history. 
It utilizes a Hybrid Architecture combining Collaborative Filtering (ALS) for candidate generation and Gradient Boosting (LightGBM) for re-ranking.

## 🚀 Live Demo
[Click here to try](https://recommendation-system-0121.streamlit.app/)<br>
Note: Service is  hosted on free-tier instances. Please wait for a minute or two.

## Architecture
```
A[User ID] --> B(Stage 1: Retrieval)
B -->|Generates Top 30 Candidates| C(Stage 2: Ranking)
C -->|Re-ranks Top 10| D[Final Recommendations]
    
"Retrieval (ALS)"
B -- Collaborative Filtering --> B1[Matrix Factorization]
end
    
"Ranking (LightGBM)"
C -- Feature Engineering --> C1[User-Item Interaction Features]
C1 --> C2[Learning to Rank]
end
```
1. **Retrieval Layer (ALS):** Uses Alternating Least Squares to quickly fetch top 30 relevant items from thousands of products based on latent user factors.<br>

2. **Ranking Layer (LightGBM):** A Gradient Boosting model re-ranks these 30 candidates by predicting the probability of interaction, using engineered features (e.g., dot product scores).

## 📊 Dataset & Performance

**Dataset: Amazon Electronics Reviews**

**Metric: Recall@30 ≈ 5%**

## ⚙️ Local Setup
**1. Clone the Repository**
```python
git clone https://github.com/YOUR_USERNAME/recommendation-system-deployment.git
cd recommendation-system-deployment
```
**2. Install Dependencies**
**Note:** Recommended to use virtual env.
```python
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-backend.txt
```
**3. Run the Backend (API)**
```python
uvicorn src.api.app:app --reload
```
**4. Run the Frontend (Dashboard)**
```python
streamlit run dashboard.py
```

## Deployment (MLOps)
**Model Versioning:** Large model artifacts (>100MB) are tracked using Git LFS.<br>
**Cloud Hosting:**
- **Render:** Automatically builds the Docker image from the main branch and deploys the API.<br>
- **Streamlit Cloud:** Hosts the frontend dashboard, connected directly to the GitHub repository.
