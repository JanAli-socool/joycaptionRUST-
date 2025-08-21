✅ Full QA Run Steps for `joycaption-candle` on RunPod
-----------------------------------------------------

### 1️⃣ Start the container

-   Use the RunPod web terminal to start the container using your image, my actual docker image --> `devtester01/joycaption-candle:gpu`.
-   Ensure the container is running with GPU enabled.
-   IP for RDP connecting with the loaded runpod is shown in image and its user/pwd is (tester / tester123A@_1)
-   start the instance and choose GPU

  <img width="537" height="293" alt="image" src="https://github.com/user-attachments/assets/e779adeb-0b12-4901-91b7-3dc41d354af8" />
  <img width="2489" height="653" alt="image" src="https://github.com/user-attachments/assets/c0d54caa-0a13-4d06-a1da-932990604623" />



* * * * *

### 2️⃣ Open the web terminal option

-   Once the container is running, click **"Web Terminal"** in RunPod.

-   You should see a prompt like:

`root@<container_id>:/workspace#`
<img width="2489" height="990" alt="image" src="https://github.com/user-attachments/assets/bce963a3-b500-4717-8b1f-395991932caa" />

* * * * *

### 3️⃣ Navigate to the project directory

`cd /workspace`   
# or 
`/app` --> depending on container mount`

-   Verify the contents:
ls

# Expected files:
Cargo.toml  Cargo.lock  src/  target/  caption-image.png

* * * * *

### 4️⃣ Set HuggingFace Hub token

-   To allow model downloads:

`export HF_HUB_TOKEN="hf_tcJoayrmMgSkrRZZBIXzijLQxCUuYcpdG"`

-   Optional: verify the token is set:

`echo $HF_HUB_TOKEN`

* * * * *

### 5️⃣ Clean HuggingFace cache

-   Prevents lock or stuck download issues:

`rm -rf /root/.cache/huggingface/hub/models--fancyfeast--llama-joycaption-beta-one-hf-llava/`

`rm -f /root/.cache/huggingface/hub/models--fancyfeast--llama-joycaption-beta-one-hf-llava/blobs/*.lock`

* * * * *

### 6️⃣ Build the Rust project (optional command already covered in ci/cd layering)

`cargo build --release (not needed)` 

-   After build, check that the binary exists:

`ls target/release/
# Should include: joycaption-candle`

* * * * *

### 7️⃣ Run 

`bash /start.sh`

for live logs :

`bash -c '/start.sh' 2>&1 | tee /workspace/joycaption_rust_output.log`

-   **Explanation**:

    -   `2>&1` → merges standard output and errors so nothing is missed.

    -   `tee` → prints logs to terminal **and** saves them to a file for later QA verification.

    -   This ensures the QA person can **watch the model download and generation process**.

* * * * *

### 8️⃣ Verify outputs

-   After execution, check the log file:

`cat /workspace/joycaption_rust_output.log`

-   Compare against expected Python outputs (visible in the colab link in same VM tab1 of edge browser).

* * * * *

### 9️⃣ Optional: remove partial caches if errors occur (very important if you are facing any  error prior to .lock)

-   If the script fails during model download:

`rm -f /root/.cache/huggingface/hub/models--fancyfeast--llama-joycaption-beta-one-hf-llava/blobs/*.lock
rm -rf /root/.cache/huggingface/hub/models--fancyfeast--llama-joycaption-beta-one-hf-llava/`

-   Then **rerun Step 7**.

* * * * *

### 🔹 Notes for QA

1.  Always use the web terminal provided by RunPod. Do **not** try `docker exec` inside the container, as Docker CLI is not available.

2.  Ensure the `HF_HUB_TOKEN` environment variable is set **before running** `start.sh`.

3.  Downloads are large (2--3 GB), so patience is required during the first run.

4.  CPU-only mode is **not supported**; keep `CPU=false` since the model is GPU-optimized.
