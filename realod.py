import cv2
import torch
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
from datetime import datetime
import pyttsx3

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Contextual info
context_info = {
    'bottle': 'Used to store liquids. Common in households, restaurants, and labs.',
    'book': 'Used for reading. Common in educational or office environments.',
    'cell phone': 'Portable device for communication and internet access.',
    'cup': 'A container used for drinking beverages.',
    "person": "A person is a human being. Humans are characterized by their ability to walk upright, use complex language, and create tools.",
    "bicycle": "A bicycle is a human-powered or motor-powered, pedal-driven vehicle with two wheels attached to a frame.",
    "car": "A car is a wheeled motor vehicle used for transportation. Most definitions state that cars run primarily on roads and have seating for one to eight people.",
    "motorcycle": "A motorcycle is a two- or three-wheeled motor vehicle. Motorcycle design varies greatly to suit a range of different purposes.",
    "airplane": "An airplane is a powered, fixed-wing aircraft that is propelled forward by thrust from a jet engine or propeller.",
    "bus": "A bus is a large motor vehicle that carries passengers by road, typically one that serves the public on a fixed route and for a fare.",
    "train": "A train is a series of connected vehicles that run on tracks and are used for transporting cargo or passengers.",
    "truck": "A truck is a motor vehicle designed to transport cargo. Trucks vary greatly in size, power, and configuration.",
    "boat": "A boat is a small vessel for traveling over water, propelled by oars, sails, or an engine.",
    "traffic light": "A traffic light is a signaling device positioned at road intersections to control the flow of traffic by using colored lights.",
    "fire hydrant": "A fire hydrant is a connection point by which firefighters can tap into a water supply for firefighting purposes.",
    "stop sign": "A stop sign is a regulatory sign used to notify drivers that they must come to a complete stop and ensure safety before proceeding.",
    "parking meter": "A parking meter is a device used to collect money in exchange for the right to park a vehicle in a particular place for a limited time.",
    "bench": "A bench is a long seat typically made of wood or metal, for multiple people to sit on, often placed in public areas like parks.",
    "bird": "Birds are warm-blooded, egg-laying creatures with feathers, wings, and beaks. Most can fly, although some are flightless.",
    "cat": "A cat is a small, carnivorous mammal that is often kept as a pet. Known for its agility and independence.",
    "dog": "A dog is a domesticated mammal that is a popular pet and working animal. Known for loyalty and companionship.",
    "horse": "A horse is a large mammal known for its strength, speed, and use in riding and work tasks.",
    "sheep": "Sheep are domesticated ruminants raised for their wool, meat, and milk. They typically have thick, woolly coats.",
    "cow": "A cow is a large domesticated ungulate raised for milk, meat, and leather. Commonly found on farms.",
    "elephant": "An elephant is the largest land animal, known for its long trunk, tusks, and intelligence. Native to Africa and Asia.",
    "bear": "A bear is a large, heavy mammal with thick fur and a short tail. Bears are omnivores and can be found in forests and mountains.",
    "zebra": "A zebra is a wild horse with black-and-white stripes found in Africa. Known for their social behavior and grazing habits.",
    "giraffe": "A giraffe is the tallest land animal, with a long neck and legs, and distinctive spots. Found in African savannas.",
    "backpack": "A backpack is a cloth sack carried on one's back, secured with two straps over the shoulders, used for carrying items.",
    "umbrella": "An umbrella is a folding canopy supported by metal ribs and mounted on a pole, used for protection from rain or sun.",
    "handbag": "A handbag is a small bag used by women to carry personal items such as money, keys, and cosmetics.",
    "tie": "A tie is a piece of cloth worn around the neck, typically by men, for formal occasions or professional dress.",
    "suitcase": "A suitcase is a rectangular case with a handle, used for carrying clothes and personal belongings while traveling.",
    "frisbee": "A frisbee is a flat, round disc typically made of plastic, used for recreational throwing and catching games.",
    "skis": "Skis are long, narrow pieces of hard material worn on the feet to glide over snow.",
    "snowboard": "A snowboard is a flat board used for sliding down snow-covered slopes.",
    "sports ball": "A sports ball is any round or oval object used in various games and sports.",
    "kite": "A kite is a light frame covered with cloth, plastic, or paper, designed to be flown in the wind at the end of a long string.",
    "baseball bat": "A baseball bat is a smooth wooden or metal club used in baseball to hit the ball.",
    "baseball glove": "A baseball glove is a large padded leather glove worn by baseball players to catch the ball.",
    "skateboard": "A skateboard is a short board mounted on wheels, ridden in a standing or crouching position.",
    "surfboard": "A surfboard is a long, narrow board used for riding waves in the sport of surfing.",
    "tennis racket": "A tennis racket is a sports implement used to hit a ball in tennis, consisting of a handled frame with an open hoop strung with cords.",
    "bottle": "A bottle is a narrow-necked container made of glass or plastic, used for storing drinks or other liquids.",
    "wine glass": "A wine glass is a type of glass used to drink wine, typically with a stem and a round bowl.",
    "cup": "A cup is a small open container used for drinking.",
    "fork": "A fork is an eating utensil with two or more prongs used to lift food to the mouth.",
    "knife": "A knife is a tool with a sharp blade used for cutting or as a weapon.",
    "spoon": "A spoon is a utensil consisting of a small shallow bowl, oval or round, at the end of a handle.",
    "bowl": "A bowl is a round, deep dish or basin used for food or liquid.",
    "banana": "A banana is a long curved fruit with a yellow skin and soft, sweet flesh.",
    "apple": "An apple is a round fruit with red, green, or yellow skin and a crisp flesh.",
    "sandwich": "A sandwich is a food consisting of one or more types of food placed between slices of bread.",
    "orange": "An orange is a citrus fruit with a tough skin and juicy, sweet flesh.",
    "broccoli": "Broccoli is an edible green plant in the cabbage family, whose large flowering head is eaten as a vegetable.",
    "carrot": "A carrot is a root vegetable, usually orange in color, though purple, black, red, white, and yellow varieties exist.",
    "hot dog": "A hot dog is a cooked sausage served in a sliced bun as a sandwich.",
    "pizza": "Pizza is a savory dish of Italian origin consisting of a usually round, flat base of dough baked with a topping of tomatoes and cheese.",
    "donut": "A donut is a type of fried dough confection or dessert food, typically ring-shaped.",
    "cake": "Cake is a sweet baked dessert, usually made from flour, sugar, and other ingredients.",
    "chair": "A chair is a piece of furniture with a raised surface used to sit on.",
    "couch": "A couch is a piece of furniture for seating two or more people in a sitting or reclining position.",
    "potted plant": "A potted plant is a plant grown in a container, commonly used indoors for decoration.",
    "bed": "A bed is a piece of furniture which is used as a place to sleep or relax.",
    "dining table": "A dining table is a table at which meals are served and eaten.",
    "toilet": "A toilet is a sanitation fixture used for the disposal of human waste.",
    "tv": "A TV (television) is an electronic device used for viewing audiovisual content.",
    "laptop": "A laptop is a portable computer with a screen and keyboard integrated into a single unit.",
    "mouse": "A mouse is a handheld pointing device used to interact with a computer.",
    "remote": "A remote is a wireless device used to control electronics like televisions and media players from a distance.",
    "keyboard": "A keyboard is an input device used to type text and interact with a computer.",
    "cell phone": "A cell phone is a mobile device used for communication, internet access, and applications.",
    "microwave": "A microwave is an electric oven that heats food using microwave radiation.",
    "oven": "An oven is a thermally insulated chamber used for the heating, baking, or drying of a substance.",
    "toaster": "A toaster is an electrical appliance designed to brown sliced bread by exposing it to radiant heat.",
    "sink": "A sink is a bowl-shaped plumbing fixture used for washing hands, dishes, and other tasks.",
    "refrigerator": "A refrigerator is an appliance used to keep food and drinks cool and fresh.",
    "book": "A book is a set of written, printed, or blank pages fastened together and enclosed between covers.",
    "clock": "A clock is a device used to measure, keep, and indicate time.",
    "vase": "A vase is a container used to hold cut flowers or for decoration.",
    "scissors": "Scissors are a hand-operated cutting instrument consisting of a pair of metal blades.",
    "teddy bear": "A teddy bear is a soft toy in the form of a bear, often given to children as a comfort object.",
    "hair drier": "A hair drier is an electric device used to dry and style hair by blowing warm air.",
    "toothbrush": "A toothbrush is a small brush used for cleaning teeth, usually with toothpaste."
}

def estimate_distance(box_height, frame_height, known_height=1.7):
    """
    Estimate distance using the height of the bounding box
    
    Parameters:
    - box_height: Height of the bounding box in pixels
    - frame_height: Height of the frame in pixels
    - known_height: Reference height of the object in meters (default is average human height)
    
    Returns: Distance in meters (approximate)
    """
    # Focal length estimation (can be calibrated more precisely if needed)
    focal_length = 500
    
    # Calculate distance
    distance = (known_height * focal_length) / box_height
    
    return distance

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
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Adjust speech rate

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
                            
                        distance = estimate_distance(box_height, frame.shape[0], known_height)
                        
                        # Draw bounding box and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Draw bounding box and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {confidence:.2f} ({distance:.1f}m)", (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),2)            
                        labels_detected.add((label, distance))
                        
                        # Voice assistant: speak detected object and position
                        frame_center = frame.shape[1] // 2
                        object_center = (x1 + x2) // 2
                        position = "left" if object_center < frame_center else "right"
                        
                        announcement = f"{label} detected on the {position}, {distance:.1f} meters away"
                        if label not in self.logged_objects:
                            self.engine.say(announcement)
                            self.engine.runAndWait()

                # Update info label and table
                if labels_detected:
                    info_text = ""
                    for label, distance in labels_detected:
                        description = context_info.get(label.lower(), "No additional information available.")
                        info_text += f"📌 {label.upper()}:\n{description}\nDistance: {distance:.1f} meters\n\n"
                        if label not in self.logged_objects:
                            current_time = datetime.now().strftime("%H:%M:%S")
                            # Add distance info to the description
                            description_with_distance = f"{description}\nDistance: {distance:.1f}m"
                            self.tree.insert('', 0, values=(
                                self.serial_no, current_time, label, description_with_distance))
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