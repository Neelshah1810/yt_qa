import requests

url = "https://www.youtube.com/watch?v=cqDQV5g7zHo"
oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
response = requests.get(oembed_url)
print(response.status_code)
if response.status_code == 200:
    data = response.json()
    print("Title:", data.get('title'))
    print("Author:", data.get('author_name'))
