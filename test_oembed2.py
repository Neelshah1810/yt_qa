import requests
import json
url = "https://www.youtube.com/watch?v=cqDQV5g7zHo"
oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
response = requests.get(oembed_url)
print(json.dumps(response.json(), indent=2))
