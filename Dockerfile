# Use an official Python runtime as a parent image
FROM tensorflow/tensorflow:2.0.0a0-py3

# Set the working directory to /app
WORKDIR /line_remover_app

# Copy the current directory contents into the container at /app
COPY . /line_remover_app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 8080


# Run app.py when the container launches
CMD ["python", "app.py"]
