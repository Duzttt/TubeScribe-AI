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

# Create temp dir
RUN mkdir -p temp && chown -R node:node /app

# Switch to non-root user
USER node

# Expose port 7860
EXPOSE 7860

# Start the server
CMD [ "npm", "start" ]