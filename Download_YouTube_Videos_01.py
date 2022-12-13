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

url = 'https://youtu.be/8Ramec1_RNU'


Download(url)