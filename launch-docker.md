# AMDGPU passthrough to docker
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host netfound:test

# Might want to use this
export TORCH_BLAS_PREFER_HIPBLASLT=1
