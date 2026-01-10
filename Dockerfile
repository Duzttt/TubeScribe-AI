# Use Node 18
FROM node:18

# Create app directory
WORKDIR /app

# Copy package files
COPY package.json ./

# Install dependencies
RUN npm install

# Copy the rest of your app source code
COPY . .

# Create the temp directory for audio downloads
# And give permission to the non-root user (node)
RUN mkdir -p temp && chown -R node:node /app

# Switch to non-root user for security (and to match folder permissions)
USER node

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Start the server
CMD [ "npm", "start" ]