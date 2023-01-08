from pytube import YouTube
from pytube import Playlist

# https://www.freecodecamp.org/news/python-program-to-download-youtube-videos/
# https://pytube.io/en/latest/api.html

def DownloadPlaylist(link, cur_dir):
    playlist = Playlist(link)
    print('Number of videos in playlist: %s' % len(playlist.video_urls))
    # playlist.download_all()
    for video in playlist.videos:
        print('downloading : {} with url : {}'.format(video.title, video.watch_url))
        video.streams. \
            filter(type='video', progressive=True, file_extension='mp4'). \
            order_by('resolution'). \
            desc(). \
            first(). \
            download(cur_dir)

def Download(link):
    youtubeObject = YouTube(link)
    youtubeObject = youtubeObject.streams.get_highest_resolution()
    # youtubeObject = youtubeObject.streams.filter(res="360p").first()

    try:
        youtubeObject.download()
    except:
        print("An error has occurred")
    print("Download is completed successfully")

# Los Puentes 2021-04-23 Day
urls = ['https://youtu.be/KpsCZ_9NDXs',
        'https://youtu.be/1Od553IyLSQ',
        'https://youtu.be/dhneV0_ueJU',
        'https://youtu.be/UWKsGx_W5G0',
        'https://youtu.be/WY4AfUkw-vM',
        'https://youtu.be/IazapbOzCtI',
        'https://youtu.be/gdjKITGhrAU',
        'https://youtu.be/kuPalXd117M',
        'https://youtu.be/oLneBZvD7OY',
        'https://youtu.be/dbPI95nzewU',
        'https://youtu.be/2x1JiY4unOk',
        'https://youtu.be/DttJv8cWW_U',
        'https://youtu.be/4MA0c0lm4NM',
        'https://youtu.be/GndjKWz0FY8']

# Los Puentes 2021-04-25 Evening
# urls = ['https://youtu.be/hXFgJInu_PU',
#         'https://youtu.be/bu_MGXWe0u0',
#         'https://youtu.be/jyZHmTpX0TA',
#         'https://youtu.be/gUH7HKClWCE',
#         'https://youtu.be/7HSYOA-7M_Y',
#         'https://youtu.be/PjxqeFIPQ1g',
#         'https://youtu.be/b-NqR10mCvI',
#         'https://youtu.be/PO4Z0MtrQl0',
#         'https://youtu.be/k0GiqdH6CD4',
#         'https://youtu.be/Yao_Hu1jScs',
#         'https://youtu.be/BKZczwVdnRE',
#         'https://youtu.be/2XpLwZ-GOxY',
#         'https://youtu.be/6cjmP6pUG90',
#         'https://youtu.be/46QWDKUr0sQ',
#         'https://youtu.be/4r1_ZdAyrM8',
#         'https://youtu.be/bJqsXwKOW9s',
#         'https://youtu.be/GW-Y0lzHoHg',
#         'https://youtu.be/cHwoUOfeDOw'
#         ]

# i = 1
# for url in urls:
#     print(f'i={i}, url={url}')
#     i += 1
#     Download(url)

# Tesoros de Kazan 2021-10-08 Day
playlistLink = 'https://www.youtube.com/playlist?list=PLq07ekK6H4qRaCGjd1QxSASS3bLe7Im7f'
cur_dir = 'video/Tesoros de Kazan 2021-10-08 Day'
DownloadPlaylist(playlistLink, cur_dir)