import yt_dlp
import sys
import os

ydl_opts = {
    'quiet': False,
    'skip_download': True,
    'js_runtimes': {'node': {}},
    'impersonate': 'chrome'
}
try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info("https://youtu.be/cqDQV5g7zHo", download=False, process=False)
        print("Success!", info.get('title'))
except Exception as e:
    print("Error:", e)
