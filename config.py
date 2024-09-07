# config.py

import os

# Directory to store uploaded files
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')

# MongoDB connection URI
MONGO_URI = 'mongodb://localhost:27017/businessAnalysis'

# Secret key for Flask sessions and flash messages
SECRET_KEY = 'password'
