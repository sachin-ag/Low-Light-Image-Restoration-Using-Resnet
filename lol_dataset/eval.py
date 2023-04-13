import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
import tensorflow as tf
from load_data import convert_to_12_channels
from model import build_model

tf.get_logger().setLevel('ERROR')

if __name__ == "__main__":

	if len(sys.argv) == 3:
		test_folder = sys.argv[1]
		result_folder = sys.argv[2]

		test_file_paths = os.listdir(test_folder)
		test_file_paths = [os.path.join(test_folder, file_path) for file_path in test_file_paths]
		test_images = [(tf.image.decode_png(tf.io.read_file(image_path), channels=3), os.path.basename(image_path)) for image_path in test_file_paths]
		test_images = [(convert_to_12_channels(image), image_filename) for (image, image_filename) in test_images]

		model = build_model()
		model.load_weights("./checkpoint/")

		pred_images = [(model.predict(tf.expand_dims(test_image, axis=0)), os.path.join(result_folder, image_filename)) for (test_image, image_filename) in test_images]
		for pred_image, file_path in pred_images:
			tf.keras.utils.save_img(file_path, tf.squeeze(pred_image), scale=True)

	else:

		print("Usage: python eval.py <path/to/test/images>  <path/to/result/images>")