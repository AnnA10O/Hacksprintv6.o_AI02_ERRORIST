import cv2
import torch
import tkinter as tk
from tkinter import ttk, scrolledtext, Canvas
from PIL import Image, ImageTk
from datetime import datetime
import pyttsx3
import time
import requests
import threading
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import queue

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Fallback context info if Wikipedia fails
context_info = {
    'bottle': 'Used to store liquids. Common in households, restaurants, and labs.',
    'book': 'Used for reading. Common in educational or office environments.',
    'cell phone': 'Portable device for communication and internet access.',
    'cup': 'A container used for drinking beverages.',
    # ... (rest of the dictionary remains the same)
}

# Color theme
bg_color = "#e0f7fa"         # Light aqua
accent_color = "#00796b"     # Teal green
text_color = "#004d40"       # Deep teal
highlight_color = "#b2ebf2"  # Pale cyan

# Cache for Wikipedia descriptions to avoid repeated API calls
wiki_cache = {}

def get_wikipedia_description(object_name, max_length=150):
    """
    Get description from Wikipedia API for the given object with improved error handling
    """
    # Check if we already have this in our cache
    if object_name.lower() in wiki_cache:
        return wiki_cache[object_name.lower()]
    
    try:
        # Construct the API URL for Wikipedia's API
        api_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + object_name.replace(" ", "_")
        
        # Make the request with a timeout
        response = requests.get(api_url, timeout=3)
        
        if response.status_code == 200:
            data = response.json()
            # Extract the description
            if 'extract' in data:
                description = data['extract']
                # Truncate if necessary
                if len(description) > max_length:
                    description = description[:max_length] + "..."
                
                # Store in cache
                wiki_cache[object_name.lower()] = description
                return description
            else:
                # Handle missing extract field
                wiki_cache[object_name.lower()] = context_info.get(object_name.lower(),  "No description available.")
                return wiki_cache[object_name.lower()]
        else:
            # Handle unsuccessful API calls
            wiki_cache[object_name.lower()] = context_info.get(object_name.lower(), "Could not retrieve information.")
            return wiki_cache[object_name.lower()]
            
    except Exception as e:
        print(f"Error fetching Wikipedia data: {e}")
        # Fall back to context info or generic message
        wiki_cache[object_name.lower()] = context_info.get(object_name.lower(), "Information temporarily unavailable.")
        return wiki_cache[object_name.lower()]
    
def estimate_distance(box_height, frame_height, known_height=0.7):
    """
    Estimate distance using the height of the bounding box
    Parameters:
    - box_height: Height of the bounding box in pixels
    - frame_height: Height of the frame in pixels
    - known_height: Reference height of the object in meters (default is average human height)
    Returns: Distance in meters (approximate)
    """
    # Focal length estimation (can be calibrated more precisely if needed)
    focal_length = 700
    
    # Calculate distance
    distance = (known_height * focal_length) / box_height
    
    return distance

class ObjectDetectionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("AI OBJECT DETECTION SYSTEM")
        self.window.geometry("1200x800")
        self.window.configure(bg=bg_color)

        #intialising log file path
        self.log_file_path = "detection_log.txt"
        self.logged_left_panel = set()
        self.left_panel_data = {}  # Dictionary to store object info for left panel
        self.max_left_panel_entries = 10  # Limit number of entries
        
        #for speech module
        self.speech_enabled = True
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background=bg_color)
        style.configure("TLabel", background=bg_color, foreground=text_color, font=("Verdana", 11))
        style.configure("Header.TLabel", font=("Orbitron", 14, "bold"), foreground=accent_color, background=bg_color)
        style.configure("Info.TLabel", font=("Consolas", 11), wraplength=300, background=bg_color, foreground=text_color)
        style.configure("Treeview", font=("Consolas", 10), background=highlight_color, fieldbackground=highlight_color, foreground=text_color)
        style.configure("Treeview.Heading", font=("Verdana", 11, "bold"), background=accent_color, foreground=bg_color)
        style.configure("Accent.TButton", background=accent_color, foreground=bg_color, font=("Verdana", 11, "bold"))

        main_container = ttk.Frame(window, padding=15)
        main_container.pack(fill=tk.BOTH, expand=True)

        self.draw_gradient_banner(main_container)

        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(content_frame, borderwidth=2, relief="groove")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        video_header = ttk.Label(left_panel, text="CAMERA FEED", style="Header.TLabel")
        video_header.pack(pady=10)
        self.video_frame = ttk.Label(left_panel)
        self.video_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        # Label for additional info
        wiki_info_label = ttk.Label(left_panel, text="OBJECT INFO", style="Header.TLabel")
        wiki_info_label.pack(pady=(5, 0))

        # Scrollable text widget to show Wikipedia/local descriptions
        self.left_info_text = scrolledtext.ScrolledText(left_panel, width=40, height=12, font=("Courier New", 10),wrap=tk.WORD, bg=highlight_color, fg=text_color, )
        self.left_info_text.pack(padx=10, pady=(0, 10), fill=tk.BOTH, expand=False)
        self.left_info_text.config(state=tk.DISABLED)


        right_panel = ttk.Frame(content_frame, width=350, borderwidth=2, relief="groove")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(0, 0))
        right_panel.pack_propagate(False)

        info_header = ttk.Label(right_panel, text="DETECTION INFORMATION", style="Header.TLabel")
        info_header.pack(pady=10)

        info_frame = ttk.Frame(right_panel)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        self.info_text = scrolledtext.ScrolledText(info_frame, width=30, height=15, font=("Courier New", 11), wrap=tk.WORD, bg=highlight_color, fg=text_color)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        self.info_text.config(state=tk.DISABLED)

        controls_frame = ttk.Frame(right_panel)
        controls_frame.pack(fill=tk.X, padx=15, pady=15)

        self.detection_status = ttk.Label(controls_frame, text="STATUS: RUNNING", foreground="green", font=("Verdana", 11, "bold"))
        self.detection_status.pack(side=tk.LEFT, pady=5)

        self.pause_button = ttk.Button(controls_frame, text="PAUSE", command=self.toggle_detection, style="Accent.TButton")
        self.pause_button.pack(side=tk.RIGHT, pady=5)

        self.speech_button = ttk.Button(controls_frame, text="MUTE SPEECH", command=self.toggle_speech, style="Accent.TButton")
        self.speech_button.pack(side=tk.RIGHT, pady=5, padx=5)
        
        log_frame = ttk.Frame(main_container, borderwidth=2, relief="groove")
        log_frame.pack(fill=tk.BOTH, pady=(15, 0))

        log_header = ttk.Label(log_frame, text="DETECTION LOG", style="Header.TLabel")
        log_header.pack(pady=10)

        table_container = ttk.Frame(log_frame)
        table_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        columns = ('#1', '#2', '#3', '#4', '#5')
        self.tree = ttk.Treeview(table_container, columns=columns, show='headings', height=8)
        self.tree.heading('#1', text='ID')
        self.tree.heading('#2', text='TIME')
        self.tree.heading('#3', text='OBJECT')
        self.tree.heading('#4', text='MOVEMENT')
        self.tree.heading('#5', text='DESCRIPTION')
        self.tree.column('#1', width=50, anchor='center')
        self.tree.column('#2', width=150)
        self.tree.column('#3', width=120)
        self.tree.column('#4', width=150)
        self.tree.column('#5', width=350)

        scrollbar = ttk.Scrollbar(table_container, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.serial_no = 1
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.paused = False
        self.logged_objects = set()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.speech_queue = queue.Queue()
        self.speech_thread = threading.Thread(target=self.process_speech_queue, daemon=True)
        self.speech_thread.start()
        
        # Object tracking dictionary to store previous positions and movement status
        self.object_tracking = {}
        self.last_announcement = {}  # Track when we last announced a status for each object
        self.announcement_cooldown = 2.0  # Seconds between announcements for the same object
        
        # Thread-safe description fetching
        self.description_queue = {}
        self.wiki_lock = threading.Lock()
        self.last_info_update_time = 0
        self.last_info_text = ""
        self.update_video()

    def update_left_panel(self, label, description, movement_status):
        """Update left panel with object information, maintaining a limited history"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] 📌 {label.upper()}\n{description}\n\n"
        
        # Create a unique key for this entry
        left_log_key = f"{label}_{movement_status}"
        
        # Check if this is a new entry
        if left_log_key not in self.left_panel_data:
            # Add to our tracking dictionary
            self.left_panel_data[left_log_key] = {
                'entry': entry,
                'timestamp': datetime.now()
            }
            
            # Limit the number of entries
            if len(self.left_panel_data) > self.max_left_panel_entries:
                # Find oldest entry
                oldest_key = min(self.left_panel_data, key=lambda k: self.left_panel_data[k]['timestamp'])
                # Remove it
                del self.left_panel_data[oldest_key]
            
            # Clear the text widget and rebuild it with current entries
            self.left_info_text.config(state=tk.NORMAL)
            self.left_info_text.delete(1.0, tk.END)
            
            # Get sorted entries (newest first)
            sorted_entries = sorted(
                self.left_panel_data.items(), 
                key=lambda x: x[1]['timestamp'], 
                reverse=True
            )
            
            # Add all current entries
            for _, entry_data in sorted_entries:
                self.left_info_text.insert(tk.END, entry_data['entry'])
                
            self.left_info_text.config(state=tk.DISABLED)
            self.left_info_text.yview(tk.END)

    #code for logging in log file
    def log_detection_to_file(self, label, status, description, distance):
        try:
            with open(self.log_file_path, "a") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_entry = f"[{timestamp}] {label.upper()} | STATUS: {status.upper()} | DISTANCE: {distance:.1f}m\nDESCRIPTION: {description}\n\n"
                f.write(log_entry)
        except Exception as e:
            print(f"Error writing to log file: {e}")

    def cleanup_old_entries(self):
        """Remove entries older than a certain threshold"""
        current_time = datetime.now()
        threshold = 60  # seconds
        
        keys_to_remove = []
        for key, data in self.left_panel_data.items():
            if (current_time - data['timestamp']).total_seconds() > threshold:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.left_panel_data[key]
        
        # Also clean up the object tracking data
        tracking_to_remove = []
        for obj_id, data in self.object_tracking.items():
            if (current_time.timestamp() - data['last_position_update']) > threshold:
                tracking_to_remove.append(obj_id)
        
        for obj_id in tracking_to_remove:
            del self.object_tracking[obj_id]
            if obj_id in self.last_announcement:
                del self.last_announcement[obj_id]

    def draw_gradient_banner(self, parent):
        canvas = Canvas(parent, height=80, width=1200, highlightthickness=0)
        canvas.pack(fill=tk.X)
        self.create_gradient(canvas, 1200, 80, "#b2ebf2", "#e0f7fa")
        canvas.create_text(600, 40, text="AI OBJECT DETECTION SYSTEM", font=("Helvetica", 24, "bold"), fill=accent_color)

    def create_gradient(self, canvas, width, height, color1, color2):
        r1, g1, b1 = canvas.winfo_rgb(color1)
        r2, g2, b2 = canvas.winfo_rgb(color2)
        r_ratio = (r2 - r1) / height
        g_ratio = (g2 - g1) / height
        b_ratio = (b2 - b1) / height
        for i in range(height):
            nr = int(r1 + (r_ratio * i))
            ng = int(g1 + (g_ratio * i))
            nb = int(b1 + (b_ratio * i))
            hex_color = f"#{nr//256:02x}{ng//256:02x}{nb//256:02x}"
            canvas.create_line(0, i, width, i, fill=hex_color)

    def update_info_text(self, text):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.insert(tk.END, text)
        self.info_text.config(state=tk.DISABLED)

    def toggle_detection(self):
        self.paused = not self.paused
        if self.paused:
            self.pause_button.config(text="RESUME")
            self.detection_status.config(text="STATUS: PAUSED", foreground="orange")
        else:
            self.pause_button.config(text="PAUSE")
            self.detection_status.config(text="STATUS: RUNNING", foreground="green")

    def toggle_speech(self):
        """Toggle speech announcements on/off"""
        self.speech_enabled = not self.speech_enabled
        if self.speech_enabled:
            self.speech_button.config(text="MUTE SPEECH")
            # Clear any backlog
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                    self.speech_queue.task_done()
                except queue.Empty:
                    break
        else:
            self.speech_button.config(text="ENABLE SPEECH")

    def get_object_description(self, object_name):
        """
        Get object description, first trying Wikipedia, then falling back to dataset
        """
        # First check if we're already fetching this description
        with self.wiki_lock:
            if object_name in self.description_queue:
                return self.description_queue[object_name]
        
        # Try Wikipedia
        wiki_desc = get_wikipedia_description(object_name)
        
        if wiki_desc:
            description = f"[WIKIPEDIA]: {wiki_desc}"
        else:
            # Fall back to the local dataset
            description = context_info.get(object_name.lower(), "NO ADDITIONAL INFORMATION AVAILABLE.")
            description = f"[LOCAL DATA]: {description}"
        
        # Store in our queue
        with self.wiki_lock:
            self.description_queue[object_name] = description
            
        return description

    def fetch_descriptions_async(self, object_name):
        """
        Fetch descriptions in background thread to avoid UI blocking
        """
        def fetch_task():
            description = self.get_object_description(object_name)
            with self.wiki_lock:
                self.description_queue[object_name] = description
                
        thread = threading.Thread(target=fetch_task)
        thread.daemon = True
        thread.start()

    def determine_movement(self, object_id, current_position, current_distance):
        """
        Determine if an object is approaching, moving away, stopped, or moving sideways
        Returns movement status and direction
        """
        current_time = time.time()
        center_x, center_y = current_position
        frame_center_x = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2
        
        # Default movement status
        movement_status = "Unknown"
        direction = "center" if abs(center_x - frame_center_x) < 100 else ("left" if center_x < frame_center_x else "right")
        
        # If this is a new object, just record its position
        if object_id not in self.object_tracking:
            self.object_tracking[object_id] = {
                'prev_position': current_position,
                'prev_distance': current_distance,
                'last_position_update': current_time,
                'stopped_count': 0,
                'is_stopped': False
            }
            movement_status = "Detected"
        else:
            # Get previous data
            prev_data = self.object_tracking[object_id]
            prev_position = prev_data['prev_position']
            prev_distance = prev_data['prev_distance']
            time_diff = current_time - prev_data['last_position_update']
            
            # Only update if enough time has passed to avoid noise
            if time_diff > 0.2:
                # Calculate position change
                delta_x = center_x - prev_position[0]
                delta_y = center_y - prev_position[1]
                distance_change = current_distance - prev_distance
                
                # Calculate magnitude of movement (in pixels)
                movement_magnitude = (delta_x*2 + delta_y*2)*0.5
                
                # Update the tracking data
                self.object_tracking[object_id]['prev_position'] = current_position
                self.object_tracking[object_id]['prev_distance'] = current_distance
                self.object_tracking[object_id]['last_position_update'] = current_time
                
                # Determine movement status
                if movement_magnitude < 15:  # Threshold for considering an object stopped
                    self.object_tracking[object_id]['stopped_count'] += 1
                    if self.object_tracking[object_id]['stopped_count'] >= 3:  # Object must be still for multiple frames
                        self.object_tracking[object_id]['is_stopped'] = True
                        movement_status = "Stopped"
                    else:
                        movement_status = "Slowing down"
                else:
                    self.object_tracking[object_id]['stopped_count'] = 0
                    self.object_tracking[object_id]['is_stopped'] = False
                    
                    # Determine approach or retreat based on distance change
                    if abs(distance_change) > 0.15:  # Threshold for significant distance change
                        if distance_change < 0:
                            movement_status = "Approaching"
                        elif distance_change > 0:
                            movement_status = "Moving away"
                    else:
                        # Moving sideways
                        if abs(delta_x) > abs(delta_y):
                            movement_status = "Moving sideways"
                            direction = "left" if delta_x < 0 else "right"
                        else:
                            movement_status = "Moving vertically"
                            direction = "up" if delta_y < 0 else "down"
        
        return movement_status, direction

    def should_announce(self, object_id, status, distance):
        """
        Improved method to determine if we should announce this status update
        - Only announce important status changes
        - Use longer cooldowns for routine updates
        - Prioritize objects based on distance and movement status
        """
        current_time = time.time()

        if object_id not in self.last_announcement:
            self.last_announcement[object_id] = {"time": 0, "status": "", "distance": 0}

        last_time = self.last_announcement[object_id]["time"]
        last_status = self.last_announcement[object_id]["status"]
        last_distance = self.last_announcement[object_id]["distance"]
        
        # Calculate distance change
        distance_change = abs(distance - last_distance)
        
        # Define announcement priority levels
        priority_statuses = ["Approaching", "Detected"]  # High priority statuses
        
        # Longer cooldown for routine updates (5 seconds)
        standard_cooldown = 5.0
        
        # Shorter cooldown for priority statuses (2 seconds)
        priority_cooldown = 2.0
        
        # Significant distance change threshold
        significant_distance_change = 1.0  # meters
        
        # Always announce new objects with no previous status
        if last_status == "":
            self.last_announcement[object_id] = {
                "time": current_time, 
                "status": status,
                "distance": distance
            }
            return True
        
        # Always announce if status changed to a priority status
        if status in priority_statuses and status != last_status:
            self.last_announcement[object_id] = {
                "time": current_time, 
                "status": status,
                "distance": distance
            }
            return True
        
        # Announce significant distance changes
        if distance_change > significant_distance_change:
            self.last_announcement[object_id] = {
                "time": current_time, 
                "status": status,
                "distance": distance
            }
            return True
            
        # For other status changes, check the appropriate cooldown
        if status != last_status:
            cooldown = priority_cooldown if status in priority_statuses else standard_cooldown
            if current_time - last_time > cooldown:
                self.last_announcement[object_id] = {
                    "time": current_time, 
                    "status": status,
                    "distance": distance
                }
                return True
        
        # For status updates with no change, use standard cooldown
        elif current_time - last_time > standard_cooldown:
            self.last_announcement[object_id] = {
                "time": current_time, 
                "status": status,
                "distance": distance
            }
            return True

        return False
        
    def process_speech_queue(self):
        """
        Process speech queue with improved handling to prevent backlogs
        """
        while True:
            try:
                # Use a timeout to periodically check if we should clear the queue
                try:
                    text = self.speech_queue.get(timeout=0.5)
                except queue.Empty:
                    # No new messages, continue the loop
                    continue
                    
                # Check if the queue is getting too large (more than 3 items)
                if self.speech_queue.qsize() > 3:
                    # Clear the queue except for the most recent item
                    while self.speech_queue.qsize() > 1:
                        try:
                            self.speech_queue.get_nowait()
                            self.speech_queue.task_done()
                        except queue.Empty:
                            break
                
                # Process the current speech item
                self.engine.say(text)
                self.engine.runAndWait()
                self.speech_queue.task_done()
                
            except RuntimeError as e:
                print(f"[Speech Error] {e}")
                # Clear the current item to avoid getting stuck
                self.speech_queue.task_done()
            except Exception as e:
                print(f"[Unexpected Speech Error] {e}")
                # Prevent the thread from crashing
                time.sleep(1)


    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def update_video(self):
        if not self.running:
            return

        if not self.paused:
            ret, frame = self.cap.read()
            if ret:
                labels_detected = set()
                results = model(frame)
                
                frame_height, frame_width = frame.shape[:2]
                frame_center_x = frame_width // 2

                info_text = ""
                
                for *box, conf, cls in results.xyxy[0]:
                    label = results.names[int(cls)]
                    confidence = float(conf)
                    if confidence > 0.5:
                        x1, y1, x2, y2 = box
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        box_height = y2 - y1
                        # Use object-specific known heights for better accuracy
                        if label.lower() == 'person':
                            known_height = 1.7  # Average human height in meters
                        elif label.lower() == 'car':
                            known_height = 1.5  # Average car height in meters
                        elif label.lower() == 'bottle':
                            known_height = 0.2  # Average bottle height in meters
                        else:
                            known_height = 0.5  # Default height
                            
                        distance = estimate_distance(box_height, frame_height, known_height)
                        
                        # Calculate center position for movement tracking
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Create a unique object ID based on label and position
                        object_id = f"{label}{int(center_x//50)}{int(center_y//50)}"
                        
                        # Determine movement status and direction
                        movement_status, direction = self.determine_movement(
                            object_id, (center_x, center_y), distance)
                        
                        # Choose color based on movement
                        if movement_status == "Approaching":
                            box_color = (0, 0, 255)  # Red for approaching
                        elif movement_status == "Stopped":
                            box_color = (255, 255, 0)  # Yellow for stopped
                        elif movement_status == "Moving away":
                            box_color = (255, 0, 0)  # Blue for moving away
                        else:
                            box_color = (0, 255, 0)  # Green for other states
                        
                        # Draw bounding box and label 
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        cv2.putText(frame, f"{label} {confidence:.2f} ({distance:.1f}m)", (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                        cv2.putText(frame, f"{movement_status}", (x1, y2 + 20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        
                        # Store info for the panel
                        labels_detected.add((label, distance, movement_status, direction))
                        
                        # Start fetching description in background if not already available
                        if label not in self.description_queue:
                            self.fetch_descriptions_async(label)
                        
                        # Get the current description (either from cache or dataset)
                        with self.wiki_lock:
                            if label in self.description_queue:
                                description = self.description_queue[label]
                            else:
                                # Use fallback until Wikipedia description is ready
                                description = context_info.get(label.lower(), "Fetching information...")
                        
                        # Add object information to info text
                        info_text += f"📌 {label.upper()}:\n{description}\nDISTANCE: {distance:.1f} METERS\nSTATUS: {movement_status.upper()} ({direction.upper()})\n\n"
                        # Append to left panel
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        entry = f"[{timestamp}] 📌 {label.upper()}\n{description}\n\n"
                        self.update_left_panel(label, description, movement_status)

                        # Voice announcement with movement information
                        if self.should_announce(object_id, movement_status, distance):
                            position_text = f"on the {direction}" if direction in ["left", "right"] else f"moving {direction}"
                            
                            # Create different announcements based on movement status
                            if movement_status == "Approaching":
                                announcement = f"{label} is approaching from the {direction}, {distance:.1f} meters away"
                            elif movement_status == "Stopped":
                                announcement = f"{label} has stopped {position_text}, {distance:.1f} meters away"
                            elif movement_status == "Moving away":
                                announcement = f"{label} is moving away {position_text}, {distance:.1f} meters away"
                            elif movement_status == "Moving sideways":
                                announcement = f"{label} is moving {direction}, {distance:.1f} meters away"
                            else:
                                announcement = f"{label} detected {position_text}, {distance:.1f} meters away"
                                
                            if self.speech_enabled:
                                self.speech_queue.put(announcement)



                        # Log to the detection table
                        current_time = datetime.now().strftime("%H:%M:%S")
                        object_key = f"{label}_{movement_status}"
                        
                        if object_key not in self.logged_objects:
                            description_with_distance = f"{description}\nDISTANCE: {distance:.1f}m"
                            self.tree.insert('', 0, values=(self.serial_no, current_time, label.upper(), movement_status.upper(), description_with_distance))
                            
                            #code to append in log file
                            self.log_detection_to_file(label, movement_status, description, distance)

                            self.serial_no += 1
                            self.logged_objects.add(object_key)
                            
                            # Limit log entries
                            if len(self.tree.get_children()) > 100:
                                self.tree.delete(self.tree.get_children()[-1])
                

                # Show info text only if there are detections
                if info_text.strip():
                    self.update_info_text(info_text)
                


                # Display frame
                display_frame = cv2.resize(frame, (640, 480))
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_frame.imgtk = imgtk
                self.video_frame.configure(image=imgtk)

        self.window.after(10, self.update_video)
        if time.time() % 30 < 0.1:  # Run approximately every 30 seconds
            self.cleanup_old_entries()

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