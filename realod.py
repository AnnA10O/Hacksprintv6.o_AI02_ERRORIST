import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import cv2
import torch
import tkinter as tk
from PIL import Image, ImageTk
import random
import requests  # For Wikipedia API

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', trust_repo=True)

# Random color generator
def generate_color():
    return tuple(random.randint(0, 255) for _ in range(3))

object_colors = {}

# Convert BGR to HEX
def bgr_to_hex(bgr):
    return "#{:02x}{:02x}{:02x}".format(bgr[2], bgr[1], bgr[0])

# Wikipedia Summary Fetcher
def fetch_wikipedia_summary(query):
    try:
        response = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("extract", "No additional info found.")
        else:
            return "No additional info found."
    except Exception:
        return "Error fetching Wikipedia info."

class ObjectDetectionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("YOLOv5 Detection")
        self.video_frame = tk.Label(window)
        self.video_frame.pack(side=tk.LEFT)

        self.info_box = tk.Text(window, width=40, height=30, font=("Arial", 11))
        self.info_box.pack(side=tk.RIGHT)

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.running = True
        self.update_video()

    def update_video(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            print("⚠ Failed to grab frame.")
            return

        results = model(frame)
        self.info_box.delete("1.0", tk.END)

        for *box, conf, cls in results.xyxy[0]:
            if conf < 0.5:
                continue
            label = results.names[int(cls)]

            if label not in object_colors:
                object_colors[label] = generate_color()

            color = object_colors[label]
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            hex_color = bgr_to_hex(color)
            tag_name = f"tag_{label}"
            self.info_box.insert(tk.END, f"{label.upper()}:\n", tag_name)

            # Always fetch info from Wikipedia
            description = fetch_wikipedia_summary(label)
            self.info_box.insert(tk.END, f"{description}\n\n", tag_name)
            self.info_box.tag_config(tag_name, foreground=hex_color)

        # Display frame
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_frame.imgtk = imgtk
        self.video_frame.config(image=imgtk)

        self.window.after(10, self.update_video)

    def stop(self):
        self.running = False
        self.cap.release()
        self.window.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop)
    root.mainloop()