import os

# Secret key for session management
SECRET_KEY = 'password'

# MongoDB configuration
MONGO_URI = "mongodb://localhost:27017/customer_segmentation"

# Upload folder path
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')