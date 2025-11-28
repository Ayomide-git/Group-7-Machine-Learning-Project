import os
from pyngrok import ngrok

# =========================
#    NGROK CONFIGURATION
# =========================

# 1. Kill previous tunnels to prevent errors
ngrok.kill()

# 2. Open a ngrok tunnel to port 8501
# Note: You might need to set your auth token if the session expires:
# ngrok.set_auth_token("YOUR_AUTHTOKEN_HERE") 
public_url = ngrok.connect(8501)

print("========================================")
print(f"Streamlit is live at: {public_url}")
print("========================================")

# 3. Run the Streamlit app using OS system commands
# We use os.system instead of '!' because this is a .py file, not a notebook
os.system("streamlit run app.py --server.port 8501 --server.enableCORS false")