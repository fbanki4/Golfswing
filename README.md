# Golfswing
# Golf Swing Analysis - Google Colab Implementation

# Install required libraries
!pip install tensorflow gym pytube opencv-python scikit-learn matplotlib moviepy mediapipe

# For MediaPipe to work properly in Colab
!apt-get update
!apt-get install -y libgl1-mesa-glx

# If you want to save files to Google Drive (recommended)
from google.colab import drive
drive.mount('/content/drive')

# Create directories
import os
DOWNLOAD_PATH = '/content/drive/MyDrive/golf_videos/'
MODEL_PATH = '/content/drive/MyDrive/golf_models/'

os.makedirs(DOWNLOAD_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# Import all needed libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import gym
from gym import spaces
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import mediapipe as mp
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
import time
import random
from pytube import YouTube, Search

# Configuration parameters
CONFIG = {
    'download_path': DOWNLOAD_PATH,
    'model_path': MODEL_PATH,
    'max_videos': 10,
    'frame_width': 224,
    'frame_height': 224,
    'pose_confidence': 0.7,
    'num_swing_clusters': 5,
    'learning_rate': 0.0001,
    'batch_size': 32,
    'epochs': 20,  # Reduced for Colab
    'gamma': 0.99,  # Discount factor for RL
}

# Modified video collector for Colab that doesn't require API key
class GolfVideoCollector:
    def __init__(self, download_path=CONFIG['download_path'], max_videos=CONFIG['max_videos']):
        self.download_path = download_path
        self.max_videos = max_videos
        
        # Create download directory if it doesn't exist
        os.makedirs(self.download_path, exist_ok=True)
        
        # Search queries to try
        self.search_queries = [
            "professional golf swing slow motion",
            "pga tour golf swing analysis",
            "golf swing technique slow motion"
        ]
    
    def search_and_download_videos(self):
        """Search for and download golf swing videos using pytube"""
        print("Searching for golf videos...")
        
        all_video_urls = []
        # Try different search queries
        for query in self.search_queries:
            try:
                print(f"Searching for: {query}")
                search = Search(query)
                # Get urls from search results
                video_urls = [video.watch_url for video in search.results]
                all_video_urls.extend(video_urls)
                print(f"Found {len(video_urls)} videos for query: {query}")
                
                # Add a small delay between searches
                time.sleep(2)
            except Exception as e:
                print(f"Error searching for {query}: {str(e)}")
        
        # Remove duplicates and limit to max_videos
        unique_video_urls = list(set(all_video_urls))
        random.shuffle(unique_video_urls)
        video_urls_to_download = unique_video_urls[:self.max_videos]
        
        print(f"Attempting to download {len(video_urls_to_download)} videos...")
        
        # Download videos
        downloaded_paths = []
        for url in video_urls_to_download:
            try:
                yt = YouTube(url)
                print(f"Downloading: {yt.title} (Length: {yt.length} seconds)")
                
                # Skip videos that are too long
                if yt.length > 180:
                    print(f"Skipping - video too long ({yt.length} seconds)")
                    continue
                
                # Get the stream
                stream = yt.streams.filter(
                    progressive=True, 
                    file_extension='mp4',
                    resolution='720p'
                ).first()
                
                # If no 720p, try lower resolution
                if not stream:
                    stream = yt.streams.filter(
                        progressive=True,
                        file_extension='mp4'
                    ).order_by('resolution').desc().first()
                
                if stream:
                    video_id = yt.video_id
                    filename = f"{video_id}.mp4"
                    output_path = stream.download(output_path=self.download_path, filename=filename)
                    downloaded_paths.append(output_path)
                    print(f"Downloaded: {output_path}")
                    
                    time.sleep(3)
                else:
                    print(f"No suitable stream found for: {yt.title}")
            except Exception as e:
                print(f"Error downloading {url}: {str(e)}")
        
        print(f"Successfully downloaded {len(downloaded_paths)} videos")
        return downloaded_paths

# Swing Detector class implementation (from the original code)
class SwingDetector:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=CONFIG['pose_confidence'],
            min_tracking_confidence=CONFIG['pose_confidence']
        )
        
    def extract_frames(self, video_path, sample_rate=5):
        """Extract frames from video at a given sample rate"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                # Resize frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(frame_rgb, (CONFIG['frame_width'], CONFIG['frame_height']))
                frames.append(resized)
            
            frame_count += 1
            
        cap.release()
        return np.array(frames)
    
    def detect_pose_in_frames(self, frames):
        """Detect human pose in frames using MediaPipe"""
        pose_data = []
        
        for frame in frames:
            results = self.pose.process(frame)
            if results.pose_landmarks:
                # Extract landmark positions
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                pose_data.append(landmarks)
            else:
                pose_data.append(None)
        
        return pose_data
    
    def detect_swings(self, video_path):
        """Detect golf swings in a video using pose change velocity"""
        frames = self.extract_frames(video_path)
        pose_data = self.detect_pose_in_frames(frames)
        
        # Filter frames where pose was detected
        valid_frames = []
        valid_pose_data = []
        
        for i, pose in enumerate(pose_data):
            if pose is not None:
                valid_frames.append(frames[i])
                valid_pose_data.append(pose)
        
        if len(valid_pose_data) < 10:
            print(f"Not enough valid pose data in {video_path}")
            return []
        
        # Calculate pose movement to detect swing
        swing_segments = []
        segment_start = 0
        in_swing = False
        
        # Convert to numpy for easier calculations
        pose_array = np.array(valid_pose_data)
        
        # Calculate movement between consecutive frames (focus on wrist and arm movement)
        # For golf swing, we focus on specific joints like wrists, elbows, and shoulders
        wrist_indices = [self.mp_pose.PoseLandmark.LEFT_WRIST.value, 
                         self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        for i in range(1, len(pose_array)):
            # Calculate movement of wrists
            wrist_movement = 0
            for idx in wrist_indices:
                if idx < len(pose_array[i]) and idx < len(pose_array[i-1]):
                    prev_pos = np.array(pose_array[i-1][idx][:3])
                    curr_pos = np.array(pose_array[i][idx][:3])
                    wrist_movement += np.linalg.norm(curr_pos - prev_pos)
            
            # Detect start of swing (high wrist movement)
            if not in_swing and wrist_movement > 0.05:
                in_swing = True
                segment_start = i
            
            # Detect end of swing (low wrist movement after a period of high movement)
            elif in_swing and wrist_movement < 0.01 and (i - segment_start) > 5:
                in_swing = False
                # Get frames corresponding to the swing
                swing_frames = valid_frames[segment_start:i]
                if len(swing_frames) >= 10:  # Only keep swings with enough frames
                    swing_segments.append(swing_frames)
        
        return swing_segments

# Feature Extractor class (simplified for Colab)
class SwingFeatureExtractor:
    def __init__(self):
        # Use pre-trained CNN as feature extractor with fewer parameters for Colab
        base_model = ResNet50(weights='imagenet', include_top=False, 
                             input_shape=(CONFIG['frame_height'], CONFIG['frame_width'], 3))
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        output = Dense(256, activation='relu')(x)
        
        self.model = Model(inputs=base_model.input, outputs=output)
    
    def extract_features(self, swing_frames):
        """Extract features from a sequence of frames using the CNN model"""
        # Normalize frames
        normalized_frames = np.array(swing_frames) / 255.0
        
        # Extract features for each frame
        features = []
        for frame in normalized_frames:
            frame_expanded = np.expand_dims(frame, axis=0)
            feature_vector = self.model.predict(frame_expanded, verbose=0)
            features.append(feature_vector.flatten())
        
        # Combine frame features to represent the entire swing
        swing_feature = np.mean(features, axis=0)
        return swing_feature

# Main function to run a simplified pipeline
def run_simplified_pipeline():
    print("Starting Golf Swing Analysis Pipeline for Colab")
    
    # Step 1: Collect videos
    collector = GolfVideoCollector()
    video_paths = collector.search_and_download_videos()
    
    if not video_paths:
        print("No videos were downloaded. Exiting.")
        return
    
    # Step 2: Detect and extract swings
    print("Detecting and extracting golf swings...")
    detector = SwingDetector()
    
    all_swings = []
    for video_path in video_paths:
        swings = detector.detect_swings(video_path)
        all_swings.extend(swings)
        print(f"Detected {len(swings)} swings in {video_path}")
    
    print(f"Total swings detected: {len(all_swings)}")
    
    if not all_swings:
        print("No swings detected. Exiting.")
        return
    
    # Step 3: Extract features from swings
    print("Extracting features from swings...")
    extractor = SwingFeatureExtractor()
    
    swing_features = []
    for swing in all_swings:
        features = extractor.extract_features(swing)
        swing_features.append(features)
    
    # Step 4: Cluster similar swing types
    print("Clustering swing types...")
    kmeans = KMeans(n_clusters=CONFIG['num_swing_clusters'], random_state=42)
    swing_clusters = kmeans.fit_predict(swing_features)
    
    # Print cluster statistics
    unique_clusters, counts = np.unique(swing_clusters, return_counts=True)
    print("Swing cluster distribution:")
    for cluster_id, count in zip(unique_clusters, counts):
        print(f"  Cluster {cluster_id}: {count} swings")
    
    # Step 5: Visualize some example swings from each cluster
    print("Saving example swings from each cluster...")
    
    for cluster_id in unique_clusters:
        # Get indices of swings in this cluster
        indices = np.where(swing_clusters == cluster_id)[0]
        
        if len(indices) > 0:
            # Take the first example
            example_idx = indices[0]
            example_swing = all_swings[example_idx]
            
            # Save a few frames from this swing
            plt.figure(figsize=(15, 5))
            plt.suptitle(f"Example Swing from Cluster {cluster_id}")
            
            # Select 5 equally spaced frames to display
            num_frames = len(example_swing)
            frame_indices = np.linspace(0, num_frames-1, 5, dtype=int)
            
            for i, frame_idx in enumerate(frame_indices):
                plt.subplot(1, 5, i+1)
                plt.imshow(example_swing[frame_idx])
                plt.title(f"Frame {frame_idx}")
                plt.axis('off')
            
            # Save the figure
            plt.savefig(f"{CONFIG['model_path']}cluster_{cluster_id}_example.png")
            plt.close()
    
    print("Analysis complete!")
    print(f"Results saved to {CONFIG['model_path']}")

# Execute the pipeline
run_simplified_pipeline()
