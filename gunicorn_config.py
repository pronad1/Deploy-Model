import os
import multiprocessing

# Bind to the PORT environment variable
bind = "0.0.0.0:" + str(os.environ.get("PORT", 5000))

# Worker configuration
workers = 1  # Use single worker to avoid memory issues
worker_class = "sync"
timeout = 300  # Increased timeout for model loading
keepalive = 2

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Memory optimization
max_requests = 100  # Restart worker after 100 requests to prevent memory leaks
max_requests_jitter = 10

# Startup
preload_app = True  # Load models before forking workers
