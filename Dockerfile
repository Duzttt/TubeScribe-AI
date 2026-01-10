# Use Node 20 (The 'trick' starts here)
FROM node:20

# Create app directory
WORKDIR /app

# Copy package files
COPY package.json ./

# Install dependencies
RUN npm install

# Copy the rest of your app source code
COPY . .

# Hugging Face Spaces run on port 7860
EXPOSE 7860

# Start the server
CMD [ "node", "index.js" ]