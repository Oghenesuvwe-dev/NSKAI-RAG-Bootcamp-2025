# Deploy to Render

## Quick Deploy Steps:

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add Render deployment files"
   git push origin main
   ```

2. **Deploy on Render:**
   - Go to [render.com](https://render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repo
   - Use these settings:
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
     - **Environment Variables:**
       - `GROQ_API_KEY` = your_groq_api_key

3. **Alternative: Use render.yaml**
   - Render will auto-detect the `render.yaml` file
   - Just add your `GROQ_API_KEY` in environment variables

Your app will be live at: `https://your-app-name.onrender.com`