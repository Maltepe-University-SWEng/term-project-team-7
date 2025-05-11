📜 Overview
The Turkish Joke Generator creates culturally coherent Turkish jokes across multiple categories (Nasreddin Hoca, Temel, and General jokes). The project combines modern natural language generation techniques with Turkish linguistic patterns to produce humorous content. What makes this project unique is the implementation and comparison of two different machine learning approaches:

Fine-tuned GPT-2 Model: A transformer-based language model adapted specifically for Turkish joke generation
Custom LSTM Model: A character-level recurrent neural network built from scratch for Turkish text generation

✨ Features

Multiple Joke Categories: Generate jokes in specific categories (Nasreddin Hoca, Temel, General) or get a random selection
Dual Model Architecture: Compare outputs from both GPT-2 and LSTM models
Quality Assurance: Advanced scoring algorithm to ensure joke quality and coherence
Similarity Detection: Ensures generated jokes are original and not duplicated from the training data
Responsive Design: User-friendly interface that works across devices
Intelligent Post-Processing: Cleanup and enhancement of generated text for improved readability

🛠️ Technologies

Backend: Python, Flask
Frontend: HTML5, CSS3, Bootstrap, JavaScript
Machine Learning: PyTorch, Transformers (Hugging Face)
Development: Git, GitHub
Testing: Python unittest
Project Management: Scrum, Trello

📋 Prerequisites

Python 3.8 or higher
pip package manager
8GB RAM (recommended for model loading)
2GB available disk space

🔧 Installation

Clone the repository
bashgit clone (https://github.com/Maltepe-University-SWEng/term-project-team-7)

cd turkish-joke-generator

Create and activate a virtual environment
bashpython -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate

Install required dependencies
bashpip install -r requirements.txt

Verify data and model directories
data/               # Should contain joke datasets
├── nasreddin.json
├── temel.json
└── genel.json

models/             # Should contain pre-trained models
├── fikra_model/    # GPT-2 fine-tuned model
└── lstm_model/     # Custom LSTM model

Start the application
bashpython app.py

Access the web interface
Open your browser and navigate to http://localhost:5000

🚀 Usage

Select model type (GPT-2 or LSTM)
Choose joke category (Nasreddin Hoca, Temel, General, or Random)
Click "Fıkra Üret" (Generate Joke) button
Read and enjoy your AI-generated Turkish joke
Click "Modelleri Karşılaştır" (Compare Models) to see output from both models simultaneously

🧪 Testing
Run the comprehensive test suite with:
bashpython -m unittest test_fikra_generator.py
The test suite includes unit tests, integration tests, and API tests to ensure system reliability.
📂 Project Structure
turkish-joke-generator/
├── data/                   # Joke datasets
├── models/                 # Pre-trained models
├── static/                 # Static assets and images
├── templates/              # HTML templates
├── app.py                  # Main Flask application
├── check_joke.py           # Similarity checking tool
└── test_fikra_generator.py # Test suite
👥 Team Members

Ömer Faruk Özer - Scrum Master & Model Developer
Furkan Aksoy - Lead Developer
Mehmet Güzel - Back-End Developer
Tamay Yazgan - QA Tester
Emre Sarı - Front-End Developer
Yaren Yıldız - Data Scientist

🌟 Acknowledgements
This project was developed as part of the Software Project Management course (SE403) at Maltepe University under the guidance of Professor Ensar Gül.
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

© 2025 Turkish Joke Generator - Maltepe University SE403 Team-7
