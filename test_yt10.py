import yt_dlp
ydl_opts = {
    'quiet': False,
    'skip_download': True,
    'js_runtimes': {'node': {}},
    'extractor_args': {'youtube': {'player_client': ['android', 'web']}}
}
try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info("https://youtu.be/cqDQV5g7zHo", download=False)
        print("Success 10!", info.get('title'))
except Exception as e:
    print("Error:", e)
