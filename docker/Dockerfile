# Use NVIDIA NGC PyTorch base image

FROM nvcr.io/nvidia/pytorch:24.05-py3

ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /workspace

RUN apt update && apt install -y \
    nodejs \
    npm \
    #ln -s /usr/bin/nodejs /usr/bin/node \
    && rm -rf /var/lib/apt/lists/*

# Install JupyterLab and common extensions
RUN pip install --upgrade pip && \
    pip install jupyterlab ipywidgets matplotlib pandas seaborn && \
    pip install pycocotools pycocoevalcap scikit-image && \
    jupyter lab --generate-config

# Configure JupyterLab to run on container start
RUN echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8888" >> /root/.jupyter/jupyter_lab_config.py

# Expose JupyterLab port
EXPOSE 8888

# Default command: start JupyterLab
CMD ["jupyter", "lab", "--allow-root"]
#CMD ["jupyter", "lab", "--allow-root", "--config=/root/.jupyter/jupyter_lab_config.py"]
