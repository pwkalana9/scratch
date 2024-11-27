# Example usage
if __name__ == "__main__":
    # Initialize the processor with a batch size of 2
    processor = InstanceSegmentationProcessor(batch_size=2)

    try:
        # Example: Adding images to the processor
        for _ in range(6):  # Simulating 6 input images
            dummy_image = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)  # Replace with real input
            processor.enqueue_image(dummy_image)

        # Retrieve and process results
        for _ in range(3):  # Expect 3 batches with batch_size=2
            batch_result = processor.dequeue_result()
            if batch_result is not None:
                print(f"Processed a batch of {len(batch_result)} images.")
                for img in batch_result:
                    print("Processed an image with shape:", img.shape)
                    # Optionally display the image (requires OpenCV's GUI functions)
                    # cv2.imshow("Output Image", img)
                    # cv2.waitKey(0)

    finally:
        # Shut down the processor properly
        processor.shutdown()