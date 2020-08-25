# Build and Run 
```
docker build -t nvidia/cuda:torch110 .
docker run -it --gpus all -v [train_data_path]:/src_data nvidia/cuda:torch110 /bin/bash
```