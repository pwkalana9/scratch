import threading
import queue
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import cv2

class InstanceSegmentationProcessor:
    def __init__(self, max_queue_size=10):
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue(maxsize=max_queue_size)
        self.predictor = self._setup_predictor()
        self.worker_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.worker_thread.start()

    @staticmethod
    def _setup_predictor():
        cfg = get_cfg()
        cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/model_final_f10217.pkl"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for predictions
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
        return DefaultPredictor(cfg)

    def _inference_worker(self):
        while True:
            try:
                image = self.input_queue.get(timeout=5)  # Wait for 5 seconds for input
                if image is None:  # Shutdown signal
                    break

                # Perform inference
                outputs = self.predictor(image)

                # Get predicted masks and overlay them on the image
                masks = outputs["instances"].pred_masks.cpu().numpy()  # Get masks
                overlaid_image = image.copy()

                for mask in masks:
                    # Create a colored mask
                    color = [0, 255, 0]  # Green mask
                    mask_colored = np.zeros_like(image, dtype=np.uint8)
                    mask_colored[mask] = color

                    # Blend the original image and mask using alpha blending
                    alpha = 0.5
                    overlaid_image = cv2.addWeighted(overlaid_image, 1, mask_colored, alpha, 0)

                # Put the output image in the output queue
                self.output_queue.put(overlaid_image)
            except queue.Empty:
                continue

    def enqueue_image(self, image: np.ndarray):
        """
        Enqueue a raw image for processing.

        Args:
            image (np.ndarray): Input image in HWC format with dtype uint8.
        """
        if self.input_queue.full():
            raise queue.Full("Input queue is full. Cannot enqueue image.")
        self.input_queue.put(image)

    def dequeue_result(self) -> np.ndarray:
        """
        Dequeue the processed image.

        Returns:
            np.ndarray: Processed image with masks superimposed, or None if the queue is empty.
        """
        try:
            return self.output_queue.get(timeout=5)  # Wait for 5 seconds for output
        except queue.Empty:
            return None

    def shutdown(self):
        """Shuts down the worker thread."""
        self.input_queue.put(None)  # Send shutdown signal
        self.worker_thread.join()

# Example usage
if __name__ == "__main__":
    processor = InstanceSegmentationProcessor()

    try:
        # Example: Adding images to the processor
        for _ in range(5):  # Simulating 5 input images
            dummy_image = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)  # Replace with real input
            processor.enqueue_image(dummy_image)

        # Retrieve and process results
        for _ in range(5):
            output_image = processor.dequeue_result()
            if output_image is not None:
                print("Processed an image with shape:", output_image.shape)
                # Optionally display the image (requires OpenCV's GUI functions)
                # cv2.imshow("Output Image", output_image)
                # cv2.waitKey(0)

    finally:
        # Shut down the processor properly
        processor.shutdown()