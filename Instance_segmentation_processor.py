import threading
import queue
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList
import cv2


class InstanceSegmentationProcessor:
    def __init__(self, batch_size=1, max_queue_size=10):
        """
        Initializes the processor with batch inference capability.

        Args:
            batch_size (int): Number of images to process in a batch.
            max_queue_size (int): Maximum size of the input and output queues.
        """
        self.batch_size = batch_size
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue(maxsize=max_queue_size)
        self.model = self._setup_model()
        self.worker_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.worker_thread.start()

    @staticmethod
    def _setup_model():
        cfg = get_cfg()
        cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/model_final_f10217.pkl"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for predictions
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

        # Build and load the model
        model = build_model(cfg)
        model.eval()
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        return model

    def _inference_worker(self):
        while True:
            try:
                batch = []
                for _ in range(self.batch_size):
                    image = self.input_queue.get(timeout=5)  # Wait for input
                    if image is None:  # Shutdown signal
                        # Put the remaining images in batch back into the queue before exiting
                        for img in batch:
                            self.input_queue.put(img)
                        return
                    batch.append(image)

                # Perform batch inference
                inputs = [{"image": torch.as_tensor(image.transpose(2, 0, 1)).float().to(self.model.device)} for image in batch]
                images = ImageList.from_tensors([x["image"] for x in inputs], self.model.backbone.size_divisibility)
                features = self.model.backbone(images.tensor)
                proposals, _ = self.model.proposal_generator(images, features)
                results, _ = self.model.roi_heads(images, features, proposals)

                # Process results and overlay masks
                processed_images = []
                for image, result in zip(batch, results):
                    instances = result["instances"].to("cpu")
                    masks = instances.pred_masks.numpy()

                    overlaid_image = image.copy()
                    for mask in masks:
                        # Create a colored mask
                        color = [0, 255, 0]  # Green mask
                        mask_colored = np.zeros_like(image, dtype=np.uint8)
                        mask_colored[mask] = color

                        # Blend the original image and mask using alpha blending
                        alpha = 0.5
                        overlaid_image = cv2.addWeighted(overlaid_image, 1, mask_colored, alpha, 0)

                    processed_images.append(overlaid_image)

                # Put the processed batch into the output queue
                self.output_queue.put(processed_images)

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

    def dequeue_result(self) -> list:
        """
        Dequeue a batch of processed images.

        Returns:
            list: List of processed images with masks superimposed, or None if the queue is empty.
        """
        try:
            return self.output_queue.get(timeout=5)  # Wait for 5 seconds for output
        except queue.Empty:
            return None

    def shutdown(self):
        """Shuts down the worker thread."""
        for _ in range(self.batch_size):  # Send shutdown signal for each slot in a batch
            self.input_queue.put(None)
        self.worker_thread.join()