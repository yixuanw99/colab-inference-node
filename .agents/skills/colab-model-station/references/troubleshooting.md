# Troubleshooting & Recovery Procedures

Runbook for diagnosing and resolving common operational anomalies in `colab-model-station`.

---

## 1. CUDA Out Of Memory (OOM) Errors

### Symptom:
`torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate X GiB...`

### Root Cause Analysis & Resolution:
1. **Multiple Engines Colliding**:
   - Check if an earlier engine is still holding VRAM:
     ```bash
     bash engine.sh status
     fuser -v /dev/nvidia*
     ```
   - Kill lingering processes:
     ```bash
     bash engine.sh diffusers stop
     bash engine.sh vllm stop
     bash engine.sh comfyui stop
     ```
2. **FLUX.1 Exceeds VRAM on L4/T4**:
   - Switch precision to FP8 or enable offloading:
     ```bash
     bash engine.sh diffusers start --model "black-forest-labs/FLUX.1-schnell" --precision fp8 --enable-cpu-offload
     ```
3. **PyTorch Memory Fragmentation**:
   - Run a clean cache release:
     ```python
     import torch
     torch.cuda.empty_cache()
     torch.cuda.ipc_collect()
     ```

---

## 2. Port Conflict (Address Already In Use)

### Symptom:
`OSError: [Errno 98] Address already in use: '0.0.0.0:8000'`

### Resolution:
1. Find PID bound to port:
   ```bash
   lsof -i :8000
   # or
   netstat -nlp | grep 8000
   ```
2. Kill specific PID or stop the registered engine:
   ```bash
   bash engine.sh diffusers stop
   # or force kill if zombie
   fuser -k 8000/tcp
   ```

---

## 3. Tailscale Connection Failures

### Symptom:
`tailscale up` hangs or errors with `failed to connect to local tailscaled`.

### Resolution:
1. Ensure Tailscale is running in **Userspace Networking Mode** (standard TUN devices are restricted in Colab containers):
   ```bash
   tailscaled --tun=userspace-networking --socks5-server=localhost:1055 --outbound-http-proxy-listen=localhost:1055 &
   ```
2. Use unattended authkey authentication:
   ```bash
   tailscale up --authkey="$TAILSCALE_AUTHKEY" --hostname="colab-node" --accept-routes
   ```
3. Verify connection:
   ```bash
   tailscale status
   tailscale ip -4
   ```

---

## 4. Hugging Face Access & Rate Limit (HTTP 401 / 403)

### Symptom:
`GatedRepoError: Access to model black-forest-labs/FLUX.1-dev is restricted and you are not in the authorized list.`

### Resolution:
1. Accept license on Hugging Face:
   - Visit https://huggingface.co/black-forest-labs/FLUX.1-dev and click **Agree and access repository**.
2. Supply Hugging Face Token:
   - In Colab notebook secrets: set `HF_TOKEN`.
   - In `.env` file:
     ```bash
     echo "HF_TOKEN=hf_your_actual_token_here" >> .env
     ```
   - Export environment variable:
     ```bash
     export HF_TOKEN="hf_your_actual_token_here"
     ```

---

## 5. Colab Kernel Disconnection & Compute Unit Drain

### Symptom:
Browser disconnected, but GPU instance continues billing compute units.

### Resolution:
1. Verify idle watchdog is running:
   ```bash
   ps aux | grep idle_watchdog
   ```
2. If watchdog was not started, start it immediately:
   ```bash
   bash engine.sh watchdog start --timeout 1800
   ```
3. To trigger immediate safe termination and unassign compute units:
   ```bash
   bash engine.sh teardown
   ```
