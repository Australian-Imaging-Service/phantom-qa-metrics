FROM ubuntu:24.04

LABEL org.opencontainers.image.title="phantomkit" \
      org.opencontainers.image.description="MRI phantom QA processing pipeline" \
      org.opencontainers.image.authors="Arkiev D'Souza <arkiev.dsouza@sydney.edu.au>" \
      org.opencontainers.image.source="https://github.com/Australian-Imaging-Service/phantomkit"

ARG DEBIAN_FRONTEND=noninteractive
ARG ANTS_VERSION=2.5.3

# ---------- Base system packages ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget unzip \
        python3 python3-pip python3-venv \
        libgomp1 \
        dc bc \
        mrtrix3 \
    && rm -rf /var/lib/apt/lists/*

# ---------- Python virtual environment ----------
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# ---------- FSL via conda environment spec ----------
# Bypasses the FSL installer (which requires sudo for system PATH setup).
# Downloads miniforge to get conda/mamba, creates the FSL env from FSL's
# own published yml spec, then removes miniforge to save space.
# Downloads ~7 GB; includes eddy, topup, flirt, and related tools.
ARG FSL_VERSION=6.0.7.22
RUN curl -fsSL \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
        -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

RUN /opt/conda/bin/conda config --set remote_max_retries 10 \
    && /opt/conda/bin/conda config --set remote_connect_timeout_secs 60 \
    && /opt/conda/bin/conda config --set remote_read_timeout_secs 600 \
    && /opt/conda/bin/mamba env create -p /opt/fsl \
        -f "https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fsl-${FSL_VERSION}_linux-64.yml" \
    && /opt/conda/bin/conda clean --all -y \
    && rm -rf /opt/conda

ENV FSLDIR=/opt/fsl \
    FSLOUTPUTTYPE=NIFTI_GZ \
    FSLMULTIFILEQUIT=TRUE
ENV PATH="${FSLDIR}/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/fsl/lib"

# ---------- ANTs pre-compiled binary ----------
RUN curl -fsSL \
        "https://github.com/ANTsX/ANTs/releases/download/v${ANTS_VERSION}/ants-${ANTS_VERSION}-ubuntu-22.04-X64-gcc.zip" \
        -o /tmp/ants.zip \
    && unzip -q /tmp/ants.zip -d /opt/ \
    && find /opt -maxdepth 3 -name "antsRegistration" 2>/dev/null \
         | head -1 | xargs dirname | xargs -I{} ln -s {} /opt/ants_bin \
    && rm /tmp/ants.zip

ENV PATH="/opt/ants_bin:${PATH}"

# ---------- dcm2niix pre-compiled binary ----------
RUN curl -fsSL \
        "https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_lnx.zip" \
        -o /tmp/dcm2niix.zip \
    && unzip -q /tmp/dcm2niix.zip -d /tmp/dcm2niix \
    && install -m 755 /tmp/dcm2niix/dcm2niix /usr/local/bin/dcm2niix \
    && rm -rf /tmp/dcm2niix.zip /tmp/dcm2niix

# Polyfill for imp module removed in Python 3.12 (needed by apt MRtrix3 scripts).
# Placed here so edits to the polyfill don't invalidate the expensive FSL layer.
COPY docker/imp_polyfill.py /usr/lib/python3/dist-packages/imp.py

# ---------- phantomkit (editable install preserves template_data lookup) ----------
WORKDIR /opt/phantomkit
COPY . .
RUN /opt/venv/bin/pip install --no-cache-dir -e .

# ---------- Runtime ----------
WORKDIR /data
VOLUME ["/data/input", "/data/output"]

ENTRYPOINT ["phantomkit"]
CMD ["--help"]
