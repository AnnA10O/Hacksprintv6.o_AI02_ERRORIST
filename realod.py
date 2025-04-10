import cv2
import torch
import tkinter as tk
from PIL import Image, ImageTk
import threading

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Contextual information
context_info = {
    'person': 'A human being. Often detected in scenes involving activity or interaction.',
    'bottle': 'Used to store liquids. Common in households, restaurants, and labs.',
    'book': 'Used for reading. Common in educational or office environments.',
    'cell phone': 'Portable device for communication and internet access.'
}

# GUI setup
class ObjectDetectionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("AI Object Detection with Info")
        self.window.geometry("900x600")
        self.video_frame = tk.Label(window)
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.info_label = tk.Label(window, text="Detected Object Info", font=("Helvetica", 14), justify=tk.LEFT, wraplength=300)
        self.info_label.pack(side=tk.RIGHT, padx=10, pady=10)

        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.update_video()

    def update_video(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            # Run detection
            results = model(frame)
            labels_detected = set()

            for *box, conf, cls in results.xyxy[0]:
                label = results.names[int(cls)]
                labels_detected.add(label)
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Update info text
            if labels_detected:
                info_text = ""
                for label in labels_detected:
                    description = context_info.get(label, "No info available.")
                    info_text += f"{label.upper()}:\n{description}\n\n"
                self.info_label.config(text=info_text)
            else:
                self.info_label.config(text="No object detected.")

            # Convert to ImageTk
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)

        self.window.after(10, self.update_video)

    def stop(self):
        self.running = False
        self.cap.release()
        self.window.quit()

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop)
    root.mainloop()
