from your_script import InstanceSegmentationProcessor

# Initialize the processor
processor = InstanceSegmentationProcessor()

# Add images to the queue
processor.enqueue_image(input_image)

# Get results
result_image = processor.dequeue_result()
if result_image is not None:
    print("Received processed image.")

# Shutdown when done
processor.shutdown()