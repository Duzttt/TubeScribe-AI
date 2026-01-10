# Use Node 18
FROM node:18

# Create app directory
WORKDIR /app

# --- CRITICAL: Install Python & FFmpeg for yt-dlp ---
RUN apt-get update && \
    apt-get install -y python3 python3-pip ffmpeg && \
    apt-get clean
# ----------------------------------------------------

# Copy package files
COPY package.json ./

# Install dependencies
RUN npm install

# Copy the rest of your app source code
COPY . .

# Create the temp directory and give permissions to 'node' user
RUN mkdir -p temp && chown -R node:node /app

# Switch to non-root user for security
USER node

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Start the server
CMD [ "npm", "start" ]