# Verify Docker locally

Run these commands in PowerShell from the `ames-housing-mlops` folder. Docker Desktop must be running with its Linux engine. This walkthrough verifies the packaged app and real housing model in manual mode. The separately running native app can still provide the full natural-language workflow.

## 1. Check the engine

```powershell
docker version
```

Both **Client** and **Server** should appear. The client sends commands; the server builds images and runs containers. Signing in to Docker Desktop alone does not prove the engine is ready.

## 2. Build the image

```powershell
docker build -t ames-capstone:local .
```

An image is the packaged application, including Python and dependencies. The final `.` means to use this folder as the build context. The Dockerfile installs the pinned application dependencies and copies the code and configuration. Secrets, the Windows virtual environment, raw data and downloaded assistant weights are excluded.

## 3. Start the container

```powershell
docker run -d --name ames-capstone-local -p 127.0.0.1:8502:8501 -e AMES_LLM_ENABLED=false ames-capstone:local
```

A container is a running instance of the image. `-d` runs it in the background. Your computer's port **8502** connects to the app's port **8501** inside the container. The address is restricted to this computer. Using 8502 keeps the native app on 8501 available.

This command is for the first launch. If the named container already exists, use `docker start ames-capstone-local` to restart it; running the creation command again will report a name conflict.

## 4. Check health and make a real prediction

```powershell
docker ps --filter name=ames-capstone-local
(Invoke-WebRequest http://127.0.0.1:8502/_stcore/health).Content
docker exec ames-capstone-local python -c "from src.serving import load_serving_model,predict; from src.interface import SAMPLE_FEATURES; model,metadata=load_serving_model(); print(predict(model,metadata,SAMPLE_FEATURES))"
```

The health response should be `ok`. The prediction command runs **inside the Linux container**, loads the actual released model and its preprocessing, verifies the artifact checksum, and predicts the full example's price. For the current model release, expect approximately **159062.25**. The model downloads on first use; it is not copied from the host's virtual environment.

Open [the Docker app](http://127.0.0.1:8502/). To verify the short manual form, enter:

- Above-ground living area: **1500** square feet.
- Ames neighborhood: **North Ames**.
- Year built: **1960**.
- Material and finish quality: **6**.

Leave the optional details unknown, review the ten defaults, check the confirmation box and choose **Confirm & estimate**. Expect approximately **$160,001**. This differs from the full example because its unknown details use training defaults. The description buttons do not invoke the language assistant in manual mode.

## 5. Stop or resume the app

```powershell
docker stop ames-capstone-local
docker start ames-capstone-local
```

Stopping preserves this container and its downloaded housing model. Rebuilding an image does not update an already-created container; create a new container from the rebuilt image when testing later code changes.

The optional `docker compose up --build` setup also puts Ollama in Docker, downloads a separate copy of Qwen, and uses CPU inference by default. That is a separate deployment path; success with this walkthrough does not establish that the full Compose stack has been tested.
