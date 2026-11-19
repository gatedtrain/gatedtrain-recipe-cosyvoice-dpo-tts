FROM docker.io/nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266
# score_tts.py's Whisper pipeline decodes audio files with the ffmpeg binary; pinned to an apt snapshot.
# The package list's file name /etc/ptc-a11-packages.txt is kept from the project's earlier name: the RUN instruction
# stays byte-identical to the one that built the published rerun's image, which launch records pin by digest.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && apt-get update --snapshot 20261029T000000Z \
 && apt-get install -y --no-install-recommends --snapshot 20261029T000000Z ffmpeg \
 && rm -rf /var/lib/apt/lists/* \
 && dpkg-query -W -f='${Package}=${Version}\n' > /etc/ptc-a11-packages.txt
