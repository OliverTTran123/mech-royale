# Build from this folder: docker build -t my-bot .
# The match runs it with no network, a read-only root (writable /tmp), 1 CPU, 512 MB, and the map at /arena.
# Pinned by digest (python:3.13-slim, Python 3.13.18) so a later rebuild gets the same image.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
WORKDIR /bot
COPY bot.py config.py nav.py mechfight.py roofs.dat ./
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
