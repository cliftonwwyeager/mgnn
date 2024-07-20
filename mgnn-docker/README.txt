# MGNN Docker Deployment

This repository contains the Docker setup for deploying the MGNN Flask application with Redis and Nginx as a reverse proxy. Follow the instructions below to build and run your containers.

## Quick Start

1. **Clone the Repository**

    ```bash
    git clone https://github.com/cliftonwwyeager/mgnn.git
    cd mgnn/mgnn-docker
    ```

2. **Build and Run Docker Containers**

    Run the following command to build and start your Docker containers:

    ```bash
    docker-compose up --build
    ```

3. **Access the Application**

    - Flask Application: `http://localhost:5000/`
    - Nginx Server (reverse proxy to Flask): `http://localhost:8080/`

## Troubleshooting Steps

### 1. Docker Compose Fails to Build

- **Problem:** Docker Compose does not build the images or exits with an error.
- **Solution:** Ensure all Dockerfiles are correctly named and located in the directory as specified in the `docker-compose.yml` file. Check for typos or errors in Dockerfile commands.

### 2. Flask Application Does Not Start

- **Problem:** The Flask application container starts but crashes or does not serve requests.
- **Solution:** 
  - Check the application logs using `docker logs <container_id>` to identify any runtime errors or missing dependencies.
  - Ensure the environment variables for the Redis connection are correctly set and that the Redis server is accessible.

### 3. Nginx Server Error

- **Problem:** Nginx server fails to start or does not forward requests to the Flask application.
- **Solution:** 
  - Verify the Nginx configuration in `nginx.conf`. Ensure that the `proxy_pass` URL matches the name of the Flask service in `docker-compose.yml`.
  - Check Nginx container logs using `docker logs <container_id>` for configuration errors or missing files.

### 4. Redis Connection Issues

- **Problem:** Flask application cannot connect to Redis.
- **Solution:** 
  - Confirm that the Redis service is up and running by checking its logs.
  - Ensure that the `REDIS_HOST`, `REDIS_PORT`, and `REDIS_PASSWORD` environment variables match those expected by your Flask application.

### 5. Performance Issues

- **Problem:** The application is slow or unresponsive.
- **Solution:** 
  - Monitor the CPU and memory usage of your containers. You can use Docker stats with `docker stats`.
  - Increase the available resources in Docker settings if necessary.

### 6. Errors After Updates

- **Problem:** Errors occur after pulling updates from the GitHub repository.
- **Solution:** 
  - Rebuild the Docker images to ensure all updates are applied: `docker-compose up --build`.
  - Check for any changes in the Docker and application configuration files.

## Reporting Issues

If you encounter any issues not covered by the troubleshooting steps, please report them by opening an issue in this GitHub repository.

## Contributing

Contributions to improve the Docker setup or application are welcome. Please fork the repository and submit a pull request with your changes.
