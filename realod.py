import cv2
import torch
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
from datetime import datetime

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Contextual info
context_info = {
    'person': 'A human being. Often detected in scenes involving activity or interaction.',
    'bottle': 'Used to store liquids. Common in households, restaurants, and labs.',
    'book': 'Used for reading. Common in educational or office environments.',
    'cell phone': 'Portable device for communication and internet access.',
    'cup': 'A container used for drinking beverages.',
}

class ObjectDetectionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Smart Object Detection")
        self.window.geometry("1200x800")
        self.window.configure(bg="#f5f5f5")

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#f5f5f5")
        style.configure("TLabel", background="#f5f5f5", font=("Helvetica", 11))
        style.configure("Header.TLabel", font=("Helvetica", 14, "bold"))
        style.configure("Info.TLabel", font=("Helvetica", 11), wraplength=300)
        style.configure("Treeview", font=("Helvetica", 10))
        style.configure("Treeview.Heading", font=("Helvetica", 11, "bold"))

        main_container = ttk.Frame(window, padding=15)
        main_container.pack(fill=tk.BOTH, expand=True)

        title_frame = ttk.Frame(main_container)
        title_frame.pack(fill=tk.X, pady=(0, 15))
        title_label = ttk.Label(title_frame, text="AI Object Detection System", font=("Helvetica", 18, "bold"))
        title_label.pack()

        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(content_frame, borderwidth=2, relief="groove")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        video_header = ttk.Label(left_panel, text="Camera Feed", style="Header.TLabel")
        video_header.pack(pady=10)
        self.video_frame = ttk.Label(left_panel)
        self.video_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        right_panel = ttk.Frame(content_frame, width=350, borderwidth=2, relief="groove")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(0, 0))
        right_panel.pack_propagate(False)

        info_header = ttk.Label(right_panel, text="Detection Information", style="Header.TLabel")
        info_header.pack(pady=10)

        info_frame = ttk.Frame(right_panel)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        self.info_text = scrolledtext.ScrolledText(info_frame, width=30, height=15, font=("Helvetica", 11), wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        self.info_text.config(state=tk.DISABLED)

        controls_frame = ttk.Frame(right_panel)
        controls_frame.pack(fill=tk.X, padx=15, pady=15)

        self.detection_status = ttk.Label(controls_frame, text="Status: Running", foreground="green", font=("Helvetica", 11, "bold"))
        self.detection_status.pack(side=tk.LEFT, pady=5)

        self.pause_button = ttk.Button(controls_frame, text="Pause", command=self.toggle_detection)
        self.pause_button.pack(side=tk.RIGHT, pady=5)

        log_frame = ttk.Frame(main_container, borderwidth=2, relief="groove")
        log_frame.pack(fill=tk.BOTH, pady=(15, 0))

        log_header = ttk.Label(log_frame, text="Detection Log", style="Header.TLabel")
        log_header.pack(pady=10)

        table_container = ttk.Frame(log_frame)
        table_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        columns = ('#1', '#2', '#3', '#4')
        self.tree = ttk.Treeview(table_container, columns=columns, show='headings', height=8)
        self.tree.heading('#1', text='ID')
        self.tree.heading('#2', text='Time')
        self.tree.heading('#3', text='Object')
        self.tree.heading('#4', text='Description')
        self.tree.column('#1', width=50, anchor='center')
        self.tree.column('#2', width=150)
        self.tree.column('#3', width=120)
        self.tree.column('#4', width=500)

        scrollbar = ttk.Scrollbar(table_container, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.serial_no = 1
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.paused = False
        self.logged_objects = set()
        self.update_video()

    def update_info_text(self, text):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, text)
        self.info_text.config(state=tk.DISABLED)

    def toggle_detection(self):
        self.paused = not self.paused
        if self.paused:
            self.pause_button.config(text="Resume")
            self.detection_status.config(text="Status: Paused", foreground="orange")
        else:
            self.pause_button.config(text="Pause")
            self.detection_status.config(text="Status: Running", foreground="green")

    def update_video(self):
        if not self.running:
            return

        if not self.paused:
            ret, frame = self.cap.read()
            if ret:
                labels_detected = set()
                results = model(frame)

                for *box, conf, cls in results.xyxy[0]:
                    label = results.names[int(cls)]
                    confidence = float(conf)
                    if confidence < 0.5:
                        continue

                    labels_detected.add(label)
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Update info label and table
                if labels_detected:
                    info_text = ""
                    for label in labels_detected:
                        description = context_info.get(label, "No additional information available.")
                        info_text += f" {label.upper()}:\n{description}\n\n"

                        if label not in self.logged_objects:
                            current_time = datetime.now().strftime("%H:%M:%S")
                            self.tree.insert('', 0, values=(
                                self.serial_no, current_time, label, description))
                            self.serial_no += 1
                            self.logged_objects.add(label)

                            if len(self.tree.get_children()) > 100:
                                self.tree.delete(self.tree.get_children()[-1])

                    self.update_info_text(info_text)
                else:
                    self.update_info_text("No objects detected in the current frame.")

                display_frame = cv2.resize(frame, (640, 480))
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_frame.imgtk = imgtk
                self.video_frame.configure(image=imgtk)

        self.window.after(10, self.update_video)

    def stop(self):
        self.running = False
        self.cap.release()
        self.window.destroy()

# Run app
if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop)
    root.mainloop()