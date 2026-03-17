# 1) Login
export CR_PAT=YOUR_GITHUB_PAT
echo "$CR_PAT" | docker login ghcr.io -u dj-codeman --password-stdin
# 2) Build
docker build -t whisper-api:latest .
# 3) Tag
docker tag whisper-api:latest ghcr.io/dj-codeman/whisper-api:v0.1.10
docker tag whisper-api:latest ghcr.io/dj-codeman/whisper-api:latest
# 4) Push
docker push ghcr.io/dj-codeman/whisper-api:v0.1.10
docker push ghcr.io/dj-codeman/whisper-api:latest