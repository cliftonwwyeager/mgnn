# Use the official NGINX image from the Docker Hub
FROM nginx:alpine

# Copy the NGINX configuration file
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose the NGINX port
EXPOSE 80