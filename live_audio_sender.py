import socket
import struct
import sys

import numpy as np
import sounddevice as sd


WSL_HOST = "172.18.72.163"
WSL_PORT = 50008

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_DURATION = 0.25
MICROPHONE_DEVICE = 18


def main():
    block_size = int(SAMPLE_RATE * BLOCK_DURATION)

    print("TinyML live microphone sender")
    print(f"Microphone device: {MICROPHONE_DEVICE}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Connecting to WSL at {WSL_HOST}:{WSL_PORT}...")

    try:
        with socket.create_connection(
            (WSL_HOST, WSL_PORT), timeout=10
        ) as connection:
            connection.settimeout(None)
            print("Connected to WSL.")
            print("Speak into your microphone.")
            print("Press Ctrl+C to stop.")

            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                device=MICROPHONE_DEVICE,
                blocksize=block_size,
            ) as stream:
                while True:
                    audio, overflowed = stream.read(block_size)

                    if overflowed:
                        print(
                            "Warning: microphone input overflowed.",
                            file=sys.stderr,
                        )

                    samples = np.asarray(
                        audio[:, 0], dtype="<f4"
                    )
                    payload = samples.tobytes()

                    # Four-byte network-order packet length,
                    # followed by float32 audio samples.
                    connection.sendall(
                        struct.pack("!I", len(payload)) + payload
                    )

    except KeyboardInterrupt:
        print("\nMicrophone sender stopped.")
    except Exception as exc:
        print(f"\nSender error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
