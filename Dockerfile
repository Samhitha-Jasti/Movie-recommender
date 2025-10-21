# Complete Dockerfile for Jupyter Notebook
FROM jupyter/base-notebook

# Copy requirements file
COPY requirements_py11s.txt /tmp/requirements_py11.txt

# Install Python packages
RUN pip install --no-cache-dir -r /tmp/requirements_py11.txt

# Copy notebook and data files
COPY --chown=${NB_UID}:${NB_GID} . /home/jovyan/work/

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter when container runs
CMD ["start-notebook.py", "--NotebookApp.token=''", "--NotebookApp.password=''"]
