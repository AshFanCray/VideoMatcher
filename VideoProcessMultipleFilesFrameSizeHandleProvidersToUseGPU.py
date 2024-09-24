import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.utils import face_align

class VideoMatcher:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            providers = ['CUDAExecutionProvider']
        else:
            self.device = torch.device("cpu")
            print("CUDA is not available. Using CPU.")
            providers = ['CPUExecutionProvider']
        
        # Force onnxruntime to use CUDA
        ort.set_default_logger_severity(3)
        self.ort_session = ort.InferenceSession("path/to/your/insightface/model.onnx", 
                                                providers=['CUDAExecutionProvider'])
        
        self.face_analyzer = FaceAnalysis(name='buffalo_l', providers=providers)
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        
        print(f"Providers used by onnxruntime: {self.ort_session.get_providers()}")

    def find_all_files(self, folder_path, extensions):
        return [os.path.join(root, file) 
                for root, _, files in os.walk(folder_path) 
                for file in files 
                if any(file.lower().endswith(ext) for ext in extensions)]

    def find_first_file(self, folder_path, extensions):
        for root, _, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    return os.path.join(root, file)
        return None

    def process_frame(self, frame, reference_embedding):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.face_analyzer.get(frame_rgb)
        
        for face in faces:
            similarity = np.dot(face.embedding, reference_embedding) / (np.linalg.norm(face.embedding) * np.linalg.norm(reference_embedding))
            if similarity > 0.5:  # Adjust this threshold as needed
                return True
        return False

    def process_video(self, video_path, reference_embedding, output_folder):
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise IOError(f"Error opening video file: {video_path}")

        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        extracted_frames_folder = os.path.join(output_folder, f"{video_name}_extracted_frames")
        matched_frames_folder = os.path.join(output_folder, f"{video_name}_matched_frames")
        os.makedirs(extracted_frames_folder, exist_ok=True)
        os.makedirs(matched_frames_folder, exist_ok=True)

        frame_count = 0
        matched_count = 0

        while True:
            ret, frame = video.read()
            if not ret:
                break

            cv2.imwrite(os.path.join(extracted_frames_folder, f"frame_{frame_count:06d}.jpg"), frame)
            
            if self.process_frame(frame, reference_embedding):
                cv2.imwrite(os.path.join(matched_frames_folder, f"matched_frame_{matched_count:06d}.jpg"), frame)
                matched_count += 1

            frame_count += 1

        video.release()

        output_video_path = os.path.join(output_folder, f"output_matched_{video_name}.mp4")
        self.create_video(matched_frames_folder, output_video_path, fps, (width, height))

        return frame_count, matched_count, output_video_path

    def create_video(self, input_folder, output_video_path, fps, size):
        images = [img for img in os.listdir(input_folder) if img.endswith(".jpg")]
        images.sort()

        if not images:
            print("No images found to create video.")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_video_path, fourcc, fps, size)

        for image in images:
            frame = cv2.imread(os.path.join(input_folder, image))
            video.write(frame)

        video.release()

    def process_single_video(self, video_path, reference_image_path, output_folder):
        try:
            print(f"Processing video: {video_path}")
            
            reference_image = cv2.imread(reference_image_path)
            reference_image_rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
            reference_faces = self.face_analyzer.get(reference_image_rgb)

            if not reference_faces:
                return f"No face found in the reference image: {reference_image_path}"

            reference_embedding = reference_faces[0].embedding

            frame_count, matched_count, output_video_path = self.process_video(video_path, reference_embedding, output_folder)

            return f"Processed {video_path}. Total frames: {frame_count}, Matched frames: {matched_count}, Output: {output_video_path}"

        except Exception as e:
            return f"An error occurred while processing {video_path}: {str(e)}"

    def process_videos(self, video_folder, reference_folder):
        video_paths = self.find_all_files(video_folder, ['.mp4', '.avi', '.mov', '.mkv'])
        if not video_paths:
            return "Error: No video files found in the specified folder."

        reference_image_path = self.find_first_file(reference_folder, ['.jpg', '.jpeg', '.png', '.bmp'])
        if not reference_image_path:
            return "Error: No image file found in the specified folder."

        output_folder = os.path.join(video_folder, "processed_outputs")
        os.makedirs(output_folder, exist_ok=True)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_single_video, video_path, reference_image_path, output_folder) 
                       for video_path in video_paths]
            results = [future.result() for future in as_completed(futures)]

        return "\n".join(results)

def start_process():
    video_folder = r"C:\Users\Amit_Local\Documents\Amit_Temp\ProcessVideo\VideoProcessor\VideoToProcess\Video"
    reference_folder = r"C:\Users\Amit_Local\Documents\Amit_Temp\ProcessVideo\VideoProcessor\VideoToProcess\Reference"
    matcher = VideoMatcher()
    return matcher.process_videos(video_folder, reference_folder)
