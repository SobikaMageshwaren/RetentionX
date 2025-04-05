# Step 1: Use an official Python runtime as a base image
FROM python:3.9-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the current directory contents into the container
COPY . /app
RUN pip install pandas scikit-learn


# Step 4: Expose the port that the app runs on
EXPOSE 5000

# Step 5: Define the command to run the app
CMD ["python", "app.py"]
