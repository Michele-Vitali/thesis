import json_numpy
import numpy as np
import requests
import settings

# This line enables the serialization of numpy arrays!
json_numpy.patch()

class OpenVLAClient:
    """
    This class consists in an HTTP client for the OpenVLA inference server.
    It sends:
        - RGB Image
        - Language instruction
        - Dataset normalization key
    
    It receives:
        - A 7D numpy action vector
    """

    def __init__(self, server_url: str = settings.server_url, unnorm_key: str = settings.unnorm_key, timeout: int = settings.timeout):
        """Initializes an OpenVLA client.

        Args:
            server_url (str, optional): The url where the server is being executed. Defaults to 'settings.server_url'.
            unnorm_key (str, optional): The un-normalization key used in dataset creation. Defaults to 'settings.unnorm_key'.
            timeout (int, optional): The maximum time our request can wait before being considered failed. Defaults to 'settings.timeout'.
        """

        # Set the class instance's variables.
        self.server_url = server_url.rstrip("/")
        self.unnorm_key = unnorm_key
        self.timeout = timeout

        self.session = requests.Session()

    def predict(self, image: np.ndarray, instruction: str) -> np.ndarray:
        """Effectively calls the OpenVLA server and passes a payload to it to then receive the predicted action from the model.

        Args:
            image (np.ndarray): The current image in a np.ndarray format.
            instruction (str): The natural language instruction.

        Raises:
            ValueError: If the image is passed in a different format rather than RGB (H, W, 3).
            ValueError: If the image is stored in a different type rather than uint8.
            ValueError: If the instruction is not a string or is empty.
            RuntimeError: If the server returns a non-valid response.
            RuntimeError: If the server returns an action in a different format rather than the expected 7D one.
            RuntimeError: If the server returns a 7D action with non-valid values in it.

        Returns:
            np.ndarray: The predicted 7D action.
        """

        # Ensure the image is an np.ndarray
        image = np.asarray(image)

        # Various checks on parameters format.
        if image.ndim != 3 or image.shape[-1] != -3:
            raise ValueError(f"An RGB image (H, W, 3) was expected, got {image.shape}.")

        if image.dtype != np.uint8:
            raise ValueError("Image must have a dtype uint8.")

        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("A non-empty language instruction is required.")

        # Build the payload to pass in the HTTP request.
        payload = {
            "image": image,
            "instruction": instruction,
            "unnorm_key": self.unnorm_key
        }

        # Make the HTTP POST request to the server with the defined payload and specified timeout
        response = self.session.post(f"{self.server_url}/act", json=payload, timeout=self.timeout)

        # If an HTTPError occured we raise it!
        response.raise_for_status()

        # If there was no error => Deserialize the server's response.
        result = response.json()

        # Check if the server's response is in the correct format.
        if isinstance(result, dict):
            if "action" not in result:
                raise RuntimeError(f"Unexpected server response: {result}")

            result = result['action']

        action = np.asarray(result, dtype=np.float32)

        if action.shape != (7,):
            raise RuntimeError(f"Expected a 7D action, got {action.shape}: {result}")

        if not np.all(np.isfinite(action)):
            raise RuntimeError(f"The OpenVLA model returned an invalid action: {action}")

        return action

    def close(self):
        """Simply closes the HTTP session.
        """

        self.session.close()