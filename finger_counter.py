"""
Advanced Webcam Finger Counter with Enhanced GUI
Uses MediaPipe Hands and OpenCV for accurate finger detection and counting.

Features:
- Highly tuned detection parameters for accuracy
- Modern, animated GUI with visual feedback
- Real-time finger state visualization
- Statistics tracking
- Smooth animations and color-coded displays
"""
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

# Landmark indices for tips, PIP (Proximal Interphalangeal), and MCP (Metacarpophalangeal) joints
TIP_IDS = [4, 8, 12, 16, 20]  # Thumb tip, Index, Middle, Ring, Pinky tips
PIP_IDS = [3, 6, 10, 14, 18]   # PIP joints
MCP_IDS = [2, 5, 9, 13, 17]    # MCP joints (for more accurate detection)

# Finger names
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

# Color scheme for GUI
COLORS = {
    'primary': (0, 150, 255),      # Orange
    'secondary': (255, 100, 0),    # Blue-orange
    'success': (0, 255, 100),      # Green
    'warning': (0, 255, 255),      # Yellow
    'info': (255, 200, 0),         # Cyan-yellow
    'background': (20, 20, 30),    # Dark blue-gray
    'text': (255, 255, 255),       # White
    'finger_active': (0, 255, 150), # Bright green
    'finger_inactive': (100, 100, 100), # Gray
}


class FingerCounterGUI:
    """Enhanced GUI class for finger counter with animations and statistics."""
    
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.history = deque(maxlen=30)  # Store last 30 counts for smoothing
        self.max_count = 0
        self.total_detections = 0
        self.start_time = time.time()
        self.animation_offset = 0
        
    def draw_gradient_background(self, frame, x, y, w, h, color1, color2, vertical=True):
        """Draw a gradient rectangle."""
        overlay = frame.copy()
        if vertical:
            for i in range(h):
                ratio = i / h
                color = tuple(int(c1 * (1 - ratio) + c2 * ratio) for c1, c2 in zip(color1, color2))
                cv2.rectangle(overlay, (x, y + i), (x + w, y + i + 1), color, -1)
        else:
            for i in range(w):
                ratio = i / w
                color = tuple(int(c1 * (1 - ratio) + c2 * ratio) for c1, c2 in zip(color1, color2))
                cv2.rectangle(overlay, (x + i, y), (x + i + 1, y + h), color, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    def draw_rounded_rect(self, frame, x, y, w, h, color, thickness=2, radius=10):
        """Draw a rounded rectangle."""
        # Draw main rectangle
        cv2.rectangle(frame, (x + radius, y), (x + w - radius, y + h), color, thickness)
        cv2.rectangle(frame, (x, y + radius), (x + w, y + h - radius), color, thickness)
        # Draw corners
        cv2.circle(frame, (x + radius, y + radius), radius, color, thickness)
        cv2.circle(frame, (x + w - radius, y + radius), radius, color, thickness)
        cv2.circle(frame, (x + radius, y + h - radius), radius, color, thickness)
        cv2.circle(frame, (x + w - radius, y + h - radius), radius, color, thickness)
    
    def draw_stat_panel(self, frame, total_count, per_hand_counts, finger_states_list):
        """Draw an attractive statistics panel."""
        panel_x, panel_y = 10, 10
        panel_w, panel_h = 280, 200
        
        # Draw panel background with gradient
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), 
                     COLORS['background'], -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Draw border
        self.draw_rounded_rect(frame, panel_x, panel_y, panel_w, panel_h, 
                              COLORS['primary'], thickness=2)
        
        # Title
        cv2.putText(frame, "FINGER COUNTER", (panel_x + 15, panel_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['primary'], 2, cv2.LINE_AA)
        
        # Total count (large and prominent)
        self.history.append(total_count)
        smoothed_count = int(np.mean(list(self.history)[-5:])) if self.history else total_count
        
        count_text = f"Total: {smoothed_count}"
        text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = panel_x + (panel_w - text_size[0]) // 2
        cv2.putText(frame, count_text, (text_x, panel_y + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS['success'], 3, cv2.LINE_AA)
        
        # Per-hand counts
        y_offset = panel_y + 100
        for idx, (label, count, _) in enumerate(per_hand_counts):
            hand_text = f"{label.capitalize()}: {count}"
            cv2.putText(frame, hand_text, (panel_x + 15, y_offset + idx * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['info'], 2, cv2.LINE_AA)
        
        # Update max count
        if smoothed_count > self.max_count:
            self.max_count = smoothed_count
        
        # Stats
        elapsed = time.time() - self.start_time
        fps_text = f"Max: {self.max_count} | Time: {int(elapsed)}s"
        cv2.putText(frame, fps_text, (panel_x + 15, panel_y + panel_h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['text'], 1, cv2.LINE_AA)
    
    def draw_finger_status(self, frame, finger_states):
        """Draw visual finger status indicators."""
        if not finger_states:
            return
        
        h, w = frame.shape[:2]
        status_x = w - 120
        status_y = 50
        
        # Draw finger status panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (status_x - 10, status_y - 10), 
                     (w - 10, status_y + 150), COLORS['background'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        cv2.putText(frame, "FINGERS", (status_x, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['primary'], 1, cv2.LINE_AA)
        
        # Draw each finger status
        finger_order = ["thumb", "index", "middle", "ring", "pinky"]
        for idx, finger_name in enumerate(finger_order):
            if finger_name in finger_states:
                is_up = finger_states[finger_name]
                y_pos = status_y + 25 + idx * 22
                color = COLORS['finger_active'] if is_up else COLORS['finger_inactive']
                
                # Draw circle indicator
                cv2.circle(frame, (status_x, y_pos), 6, color, -1)
                cv2.circle(frame, (status_x, y_pos), 6, COLORS['text'], 1)
                
                # Draw finger name
                display_name = finger_name.capitalize()[:5]
                cv2.putText(frame, display_name, 
                           (status_x + 15, y_pos + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    
    def draw_animated_cursor(self, frame, hand_landmarks, color=(0, 255, 255)):
        """Draw an animated cursor at the index fingertip."""
        h, w, _ = frame.shape
        index_tip = hand_landmarks.landmark[TIP_IDS[1]]
        cx, cy = int(index_tip.x * w), int(index_tip.y * h)
        
        # Animated pulsing circle
        self.animation_offset += 0.1
        radius = int(8 + 3 * np.sin(self.animation_offset))
        
        # Outer glow
        cv2.circle(frame, (cx, cy), radius + 4, color, 2)
        # Main circle
        cv2.circle(frame, (cx, cy), radius, color, -1)
        # Inner highlight
        cv2.circle(frame, (cx - 2, cy - 2), radius // 2, (255, 255, 255), -1)
        
        # Label with background
        label = "CURSOR"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_x, label_y = cx + 12, cy - 8
        
        # Background for text
        cv2.rectangle(frame, (label_x - 2, label_y - text_size[1] - 2),
                     (label_x + text_size[0] + 2, label_y + 2),
                     COLORS['background'], -1)
        
        cv2.putText(frame, label, (label_x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    def draw_instructions(self, frame):
        """Draw user instructions."""
        h, w = frame.shape[:2]
        instructions = [
            "Place your hand in front of the camera",
            "Open or close fingers to count them",
            "The dot shows your index finger cursor",
            "Press 'Q' or ESC to quit"
        ]
        
        y_start = h - 100
        for i, line in enumerate(instructions):
            y_pos = y_start + i * 20
            # Shadow effect
            cv2.putText(frame, line, (23, y_pos + 1),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            # Main text
            cv2.putText(frame, line, (22, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['text'], 1, cv2.LINE_AA)


def count_fingers(hand_landmarks, handedness_label, mirrored=False):
    """
    Count extended fingers with improved accuracy using multiple joint comparisons.
    
    Uses both PIP and MCP joints for better detection accuracy.
    Thumb uses x-axis comparison with proper handedness handling.
    Other fingers use both y-position and angle checks.
    
    Returns:
        count (int): number of extended fingers (0-5)
        states (dict): per-finger boolean states
    """
    landmarks = hand_landmarks.landmark
    
    # Get wrist position for reference
    wrist = landmarks[0]
    
    # Thumb detection - more accurate using multiple points
    thumb_tip = landmarks[TIP_IDS[0]]
    thumb_ip = landmarks[2]  # Interphalangeal joint
    thumb_mcp = landmarks[MCP_IDS[0]]
    
    label = handedness_label.lower()
    if mirrored:
        label = "left" if label.startswith("right") else "right"
    
    # Thumb is extended if tip is further from wrist than MCP joint
    if label.startswith("right"):
        thumb_up = thumb_tip.x > thumb_mcp.x
    else:
        thumb_up = thumb_tip.x < thumb_mcp.x
    
    # Additional check: thumb tip should be above MCP for extended thumb
    if thumb_up:
        thumb_up = thumb_tip.y < thumb_mcp.y + 0.05
    
    finger_states = {"thumb": thumb_up}
    
    # Other four fingers: use both PIP and MCP for accuracy
    finger_names = ["index", "middle", "ring", "pinky"]
    for name, tip_id, pip_id, mcp_id in zip(finger_names, TIP_IDS[1:], PIP_IDS[1:], MCP_IDS[1:]):
        tip = landmarks[tip_id]
        pip = landmarks[pip_id]
        mcp = landmarks[mcp_id]
        
        # Finger is extended if:
        # 1. Tip is above PIP (tip.y < pip.y)
        # 2. Tip is above MCP (tip.y < mcp.y)
        # 3. Distance from tip to wrist is greater than MCP to wrist
        tip_above_pip = tip.y < pip.y
        tip_above_mcp = tip.y < mcp.y
        
        # Calculate distances from wrist
        tip_dist = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
        mcp_dist = np.sqrt((mcp.x - wrist.x)**2 + (mcp.y - wrist.y)**2)
        
        # Finger is extended if tip is further from wrist than MCP
        extended = tip_above_pip and tip_above_mcp and (tip_dist > mcp_dist * 0.9)
        
        finger_states[name] = extended
    
    count = sum(finger_states.values())
    return count, finger_states


def draw_hand_landmarks_enhanced(frame, hand_landmarks, mp_hands, mp_draw, draw_styles):
    """Draw hand landmarks with enhanced styling."""
    # Draw connections
    mp_draw.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        draw_styles.get_default_hand_landmarks_style(),
        draw_styles.get_default_hand_connections_style(),
    )
    
    # Highlight fingertips
    h, w = frame.shape[:2]
    for tip_id in TIP_IDS:
        landmark = hand_landmarks.landmark[tip_id]
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (cx, cy), 5, COLORS['warning'], -1)
        cv2.circle(frame, (cx, cy), 5, COLORS['text'], 1)


def main():
    """Main function with enhanced GUI and tuned parameters."""
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    draw_styles = mp.solutions.drawing_styles
    
    # Try to open webcam with multiple indices
    cap = None
    for camera_idx in range(3):
        cap = cv2.VideoCapture(camera_idx)
        if cap.isOpened():
            # Set optimal resolution for balance between quality and performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            print(f"Webcam opened successfully at index {camera_idx}")
            break
        else:
            if cap:
                cap.release()
    
    if cap is None or not cap.isOpened():
        print("Error: Could not open webcam. Please check your camera connection.")
        return
    
    # Get actual frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize GUI
    gui = FingerCounterGUI(frame_width, frame_height)
    
    # Configure MediaPipe Hands with TUNED parameters for better accuracy
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,      # Increased from 0.5 for better accuracy
        min_tracking_confidence=0.7,       # Increased from 0.5 for smoother tracking
        model_complexity=1                  # Use full model for better accuracy
    ) as hands:
        print("Starting Enhanced Finger Counter...")
        print("Press 'Q' or ESC to quit.")
        
        mirror_view = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from webcam.")
                break
            
            # Flip frame horizontally for mirror effect
            if mirror_view:
                frame = cv2.flip(frame, 1)
            
            # Process frame: convert BGR to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            
            total_count = 0
            per_hand_counts = []
            finger_states_list = []
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_idx, (hand_landmarks, handedness) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)
                ):
                    label = handedness.classification[0].label
                    confidence = handedness.classification[0].score
                    
                    # Count fingers with improved accuracy
                    count, finger_states = count_fingers(hand_landmarks, label, mirrored=mirror_view)
                    total_count += count
                    per_hand_counts.append((label, count, hand_landmarks))
                    finger_states_list.append(finger_states)
                    
                    # Draw enhanced hand landmarks
                    draw_hand_landmarks_enhanced(frame, hand_landmarks, mp_hands, mp_draw, draw_styles)
                    
                    # Draw animated cursor for first hand
                    if hand_idx == 0:
                        gui.draw_animated_cursor(frame, hand_landmarks, COLORS['warning'])
            
            # Draw GUI elements
            gui.draw_stat_panel(frame, total_count, per_hand_counts, finger_states_list)
            
            if finger_states_list:
                gui.draw_finger_status(frame, finger_states_list[0])
            
            gui.draw_instructions(frame)
            
            # Display the frame
            cv2.imshow("Enhanced Finger Counter", frame)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                print("Quitting...")
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Program ended.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
