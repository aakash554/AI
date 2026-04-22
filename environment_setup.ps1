Write-Host "Setting up the Exam Revision Assistant Environment..."

# Upgrade pip
python -m pip install --upgrade pip

# Install Python requirements
Write-Host "Installing Python dependencies..."
pip install -r requirements.txt

# Ensure Ollama is running and has the models pulled
Write-Host "Pulling Llama 3.1 8B model (this might take a while if not already downloaded)..."
ollama pull llama3.1:8b

Write-Host "Pulling Nomic Embed Text model..."
ollama pull nomic-embed-text

Write-Host "Environment setup complete! Run 'streamlit run app.py' to start the app."
