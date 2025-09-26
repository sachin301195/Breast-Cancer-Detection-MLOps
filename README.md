# End-to-End MLOps: Breast Cancer Classifier

This project is a complete, production-grade demonstration of an MLOps pipeline for a deep learning-based breast cancer classifier. The system is deployed on Google Cloud as a scalable, serverless application, featuring a decoupled frontend and backend, all managed through automated CI/CD workflows.

**Live Demo:** [https://breast-cancer-demo-frontend-120059375610.northamerica-northeast2.run.app/](https://breast-cancer-demo-frontend-120059375610.northamerica-northeast2.run.app/)

---

## Project Overview

The core of this project is a binary classification model trained to distinguish between benign and malignant tumors from histopathology images. However, the primary focus is on the **MLOps and cloud engineering** required to serve this model as a reliable, scalable, and automated service.

### Key Features:
* **Interactive Frontend**: A modern, responsive web interface built with **Next.js** and **React**, allowing users to upload an image or select a sample for classification.
* **High-Performance Backend**: A **FastAPI** application serves the trained **Keras/TensorFlow** model, providing a fast and efficient REST API for predictions.
* **Serverless Deployment**: Both the frontend and backend are containerized with **Docker** and deployed as independent services on **Google Cloud Run**, enabling automatic scaling (including to zero) to handle traffic efficiently.
* **Automated CI/CD**: The entire deployment process is automated using **GitHub Actions**. Any push to the `main` branch triggers a workflow that builds the Docker images, pushes them to **Google Artifact Registry**, and deploys the new versions to Cloud Run, ensuring seamless and error-free updates.

---

## MLOps Architecture

The architecture is designed to reflect best practices in MLOps, with a clear separation of concerns between the frontend, backend, and the CI/CD pipeline.



1.  **Code & Version Control**: The source code for the frontend and backend is hosted on GitHub, with version control managed by Git.
2.  **CI/CD Pipeline (GitHub Actions)**:
    * **Trigger**: The pipeline is automatically triggered on a `git push` to the `main` branch.
    * **Build**: Two parallel jobs build Docker images for the frontend (Next.js) and backend (FastAPI) applications.
    * **Store**: The newly built Docker images are tagged with the commit SHA and pushed to Google Artifact Registry.
    * **Deploy**: The GitHub Actions workflow authenticates to Google Cloud and deploys the new images to their respective Cloud Run services, creating new revisions with zero downtime.
3.  **Serving Infrastructure (Google Cloud Run)**:
    * **Frontend Service**: Runs the Next.js container, serving the user interface to the public.
    * **Backend Service**: Runs the FastAPI container, exposing the `/predict` endpoint for the model. The frontend communicates with the backend via a secure HTTPS connection.

---

## Tech Stack

* **Machine Learning**: Keras, TensorFlow, Scikit-learn
* **Backend**: Python, FastAPI, Uvicorn
* **Frontend**: Next.js, React, TypeScript, Tailwind CSS
* **Cloud & DevOps**: Google Cloud Run, Google Artifact Registry, GitHub Actions, Docker

---

## How to Run Locally

To run this application on your local machine, you will need to run the frontend and backend services separately.

### Backend (FastAPI)
1.  Navigate to the backend directory.
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Uvicorn server:
    ```bash
    uvicorn app.main:app --reload
    ```
    The backend API will be available at `http://localhost:8000`.

### Frontend (Next.js)
1.  Navigate to the `breast-cancer-demo` directory.
2.  Install the Node.js dependencies:
    ```bash
    npm install
    ```
3.  Run the development server:
    ```bash
    npm run dev
    ```
    The frontend application will be available at `http://localhost:3000`.

---
## Contact

* **GitHub:** [https://github.com/sachin301195](https://github.com/sachin301195)
* **LinkedIn:** [https://www.linkedin.com/in/sachin-bulchandani/](https://www.linkedin.com/in/sachin-bulchandani/)
* **Email:** sachinbulchandani1@gmail.com
