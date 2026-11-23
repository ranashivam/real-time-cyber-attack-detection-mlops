
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from app.model_loader import load_model_bundle, predict_anomaly


HOST = "0.0.0.0"
PORT = 8080

model_bundle = None


# Handle health checks and prediction requests from HTTP clients.
class InferenceRequestHandler(BaseHTTPRequestHandler):

    # Send a JSON response with the appropriate HTTP status code.
    def send_json_response(self, status_code, response_data):
        response_body = json.dumps(response_data).encode("utf-8")

        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()

        self.wfile.write(response_body)

    # Respond to health checks from Docker and Kubernetes.
    def do_GET(self):
        if self.path == "/health":
            self.send_json_response(
                200,
                {
                    "status": "healthy",
                    "model_loaded": model_bundle is not None
                }
            )
            return

        self.send_json_response(
            404,
            {"error": "Endpoint not found."}
        )

    # Accept input features and return the model prediction.
    def do_POST(self):
        if self.path != "/predict":
            self.send_json_response(
                404,
                {"error": "Endpoint not found."}
            )
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))

            if content_length <= 0 or content_length > 10240:
                raise ValueError("Request body is empty or too large.")

            request_body = self.rfile.read(content_length)
            request_data = json.loads(request_body.decode("utf-8"))

            if not isinstance(request_data, dict):
                raise ValueError("Request body must be a JSON object.")

            prediction = predict_anomaly(
                model_bundle,
                request_data.get("features")
            )

            self.send_json_response(200, prediction)

        except (ValueError, json.JSONDecodeError) as error:
            self.send_json_response(
                400,
                {"error": str(error)}
            )

        except Exception:
            self.send_json_response(
                500,
                {"error": "Prediction failed."}
            )

    # Suppress default HTTP server access messages.
    def log_message(self, format_string, *args):
        print(f"[HTTP] {format_string % args}")


# Load the model and start the inference HTTP server.
def main():
    global model_bundle

    print("Loading approved Autoencoder model...")

    model_bundle = load_model_bundle()

    print("Model loaded successfully.")
    print(f"Features: {model_bundle['features']}")
    print(f"Threshold: {model_bundle['threshold']}")
    print(f"Starting inference server on {HOST}:{PORT}")

    server = HTTPServer(
        (HOST, PORT),
        InferenceRequestHandler
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down inference server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
