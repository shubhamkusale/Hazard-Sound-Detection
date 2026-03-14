# src/alert.py

import sys
import os
import platform

from src.predict import predict_sound

HAZARD_CLASSES = ["gun_shot", "siren", "drilling"]


def beep():
    if platform.system() == "Windows":
        try:
            import winsound
            winsound.Beep(1000, 500)
        except Exception:
            print("\a")
    else:
        os.system('printf "\\a"')


def trigger_alert(class_name, confidence):
    confidence_int = int(round(confidence))

    if class_name in HAZARD_CLASSES:
        print("╔══════════════════════════════╗")
        print("║  ⚠  HAZARD DETECTED         ║")
        print(f"║  Sound: {class_name:<18}║")
        print(f"║  Confidence: {confidence_int}%{' ' * (12 - len(str(confidence_int)))}║")
        print("╚══════════════════════════════╝")
        beep()
    else:
        print(f"✓ Safe sound detected: {class_name} ({confidence_int}%)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/alert.py path/to/audio.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    class_name, confidence = predict_sound(audio_path)
    trigger_alert(class_name, confidence)


if __name__ == "__main__":
    main()