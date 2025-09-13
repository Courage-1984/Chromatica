# Exposing Chromatica API Externally

## Overview

This guide explains how to expose your local Chromatica color search engine (running at `http://127.0.0.1:8000/`) to external users over the internet, allowing friends or colleagues to test the application remotely.

## Recommended Methods

### 1. ngrok (Recommended for Quick Sharing)

**Pros:**

- Simple setup and immediate results
- Secure HTTPS tunnel
- Free tier available
- Popular and well-documented

**Cons:**

- Free URLs change on restart
- Session time limits on free tier

**Setup Steps:**

1. **Download and Install:**

   ```powershell
   # Download from https://ngrok.com/download
   # Extract ngrok.exe to C:\ngrok\
   ```

2. **Create Account and Get Auth Token:**

   - Sign up at https://dashboard.ngrok.com/
   - Copy your auth token from the dashboard

3. **Authenticate:**

   ```powershell
   cd C:\ngrok
   .\ngrok.exe authtoken YOUR_AUTH_TOKEN_HERE
   ```

4. **Start Chromatica Server:**

   ```powershell
   # Activate virtual environment
   venv311\Scripts\activate

   # Navigate to API directory
   cd src\chromatica\api

   # Start the server
   python main.py
   ```

5. **Create Tunnel:**

   ```powershell
   # In a new terminal
   cd C:\ngrok
   .\ngrok.exe http 8000
   ```

6. **Share the URL:**
   - ngrok displays: `https://abc123.ngrok.io`
   - Send this URL to your friend

### 2. Cloudflare Tunnel (More Secure)

**Pros:**

- Enterprise-grade security
- Stable URLs
- No session limits
- Free tier available

**Cons:**

- More complex setup
- Requires Cloudflare account

**Setup Steps:**

1. **Install cloudflared:**

   ```powershell
   # Using Chocolatey
   choco install cloudflared

   # Or download from GitHub releases
   ```

2. **Login:**

   ```powershell
   cloudflared tunnel login
   ```

3. **Create Tunnel:**
   ```powershell
   cloudflared tunnel --url http://localhost:8000
   ```

### 3. Localtunnel (Simple Alternative)

**Pros:**

- No registration required
- Simple npm installation
- Quick setup

**Cons:**

- Random URLs
- Less security features

**Setup Steps:**

1. **Install:**

   ```powershell
   npm install -g localtunnel
   ```

2. **Create Tunnel:**
   ```powershell
   lt --port 8000
   ```

## Security Considerations

### ⚠️ Important Security Notes

1. **Public Access:** Your API becomes publicly accessible
2. **No Authentication:** Anyone with the URL can use your service
3. **Resource Usage:** External users can consume your bandwidth and processing power
4. **Data Exposure:** All API endpoints become accessible

### 🛡️ Security Best Practices

1. **Monitor Usage:**

   - Check tunnel provider dashboard for traffic
   - Monitor server logs for unusual activity

2. **Rate Limiting:**

   - Consider implementing rate limits in your FastAPI app
   - Monitor for potential abuse

3. **Temporary Sharing:**

   - Close tunnels when not needed
   - Don't leave them running indefinitely

4. **Safe for Demo:**
   - Chromatica demo is safe to share
   - No sensitive data or user information exposed

## What External Users Can Access

### Web Interface Features

- Complete color search functionality
- All 6 visualization tools:
  - Color Palette Analyzer
  - Search Results Analyzer
  - Interactive Color Explorer
  - Histogram Analysis Tool
  - Distance Debugger Tool
  - Query Visualizer Tool
- Image upload and processing
- Real-time search results

### API Endpoints

- `GET /search` - Color search functionality
- `POST /upload` - Image upload (if implemented)
- All static assets and documentation

## Troubleshooting

### Common Issues

1. **Port Conflicts:**

   ```powershell
   # Check if port 8000 is in use
   netstat -an | findstr :8000
   ```

2. **Firewall Permissions:**

   - Windows may ask for firewall permission
   - Allow access for both Python and tunnel tool

3. **URL Changes:**

   - Free ngrok URLs change on restart
   - Consider paid plan for stable URLs

4. **Performance:**
   - International access may be slower
   - Monitor server performance

### Performance Considerations

- **Bandwidth:** Image uploads consume bandwidth
- **Processing:** Color analysis is CPU-intensive
- **Concurrent Users:** Multiple users may impact performance

## Best Practices for Sharing

1. **Test First:**

   - Verify the tunnel works locally
   - Test all major features

2. **Provide Instructions:**

   - Share the URL with usage instructions
   - Explain the color search functionality

3. **Monitor Usage:**

   - Keep an eye on server performance
   - Be ready to restart if needed

4. **Clean Shutdown:**
   - Close tunnel when done
   - Stop the server properly

## Example Usage Instructions for Friends

```
Hi! I've set up a color search engine demo for you to try:

URL: https://abc123.ngrok.io

Features to try:
1. Upload an image to find similar colors
2. Use the color palette analyzer
3. Try the interactive color explorer
4. Test the search results analyzer

The system analyzes images in CIE Lab color space and finds
visually similar images based on color palettes.

Let me know what you think!
```

## Conclusion

Exposing your Chromatica API externally is straightforward using tools like ngrok. The demo is safe to share and provides a great way to showcase your color search engine to friends and colleagues. Remember to monitor usage and close tunnels when not needed.
