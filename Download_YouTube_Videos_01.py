from pytube import YouTube

# https://www.freecodecamp.org/news/python-program-to-download-youtube-videos/
# https://pytube.io/en/latest/api.html

def Download(link):
    youtubeObject = YouTube(link)
    youtubeObject = youtubeObject.streams.get_highest_resolution()
    # youtubeObject = youtubeObject.streams.filter(res="360p").first()

    try:
        youtubeObject.download()
    except:
        print("An error has occurred")
    print("Download is completed successfully")
# Los Puentes 2021-04-23 Evening
# https://www.youtube.com/watch?v=ye9U569ZqEU&list=PLq07ekK6H4qT8J-fa6iys81GXsyE61qa-&index=16
# url = 'https://youtu.be/lDDKSuFl2g4'
# url = 'https://youtu.be/_akc1q1LIwI'

# Los Puentes 2021 part 066 milonga
url = 'https://youtu.be/8cCU-z6YSJk'
# url = 'https://youtu.be/oG1B0kh4z9w'
# url = 'https://youtu.be/xTNnRVkjbrg'
# url = 'https://youtu.be/LfHHzWGkavU'
# url = 'https://youtu.be/MNCStpWu-x0'
# url = 'https://youtu.be/M9mMAhuPJOI'
# url = 'https://youtu.be/M9mMAhuPJOI'
# url = 'https://youtu.be/3tn8cB7GdDk'
# url = 'https://youtu.be/j9W7DMpcsT8'
# url = 'https://youtu.be/YtDfGhWnrmU'
# url = 'https://youtu.be/VB5LffbDva0'
# url = 'https://youtu.be/b0qK5AZ9MyE'
# url = 'https://youtu.be/pdU7JLBQu8Y'
# url = 'https://youtu.be/nEXW0wKLMSQ'
# url = 'https://youtu.be/ye9U569ZqEU'

Download(url)