from locust import HttpUser, task, between

class LoadTest(HttpUser):
    wait_time = between(1, 2)  # Random wait between 1-2 seconds

    @task
    def upload_image(self):
        with open("uploaded/template/template01.jpeg", "rb") as file1, open("uploaded/template/template01.jpeg", "rb") as file2:
            files = {
                "klaim": ("template01.jpeg", file1, "image/jpeg"),
                "pengkinian": ("template01.jpeg", file2, "image/jpeg")
            }
            response = self.client.post("/image", files=files)

            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
