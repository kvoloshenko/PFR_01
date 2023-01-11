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
# urls = ['https://youtu.be/KpsCZ_9NDXs',
#         'https://youtu.be/1Od553IyLSQ',
#         'https://youtu.be/dhneV0_ueJU',
#         'https://youtu.be/UWKsGx_W5G0',
#         'https://youtu.be/WY4AfUkw-vM',
#         'https://youtu.be/IazapbOzCtI',
#         'https://youtu.be/gdjKITGhrAU',
#         'https://youtu.be/kuPalXd117M',
#         'https://youtu.be/oLneBZvD7OY',
#         'https://youtu.be/dbPI95nzewU',
#         'https://youtu.be/2x1JiY4unOk',
#         'https://youtu.be/DttJv8cWW_U',
#         'https://youtu.be/4MA0c0lm4NM',
#         'https://youtu.be/GndjKWz0FY8']

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

# Tesoros de Kazan 2021-10-08 Day
urls = ['https://youtube.com/watch?v=xXEJ_3_eui0',
'https://youtube.com/watch?v=At8_CA4wP_c',
'https://youtube.com/watch?v=n99V1UHjguE',
'https://youtube.com/watch?v=obmVvmYJBkU',
'https://youtube.com/watch?v=Uenmkh__XB8',
'https://youtube.com/watch?v=to1PTR15RxM',
'https://youtube.com/watch?v=7JQyb8AKeQ4',
'https://youtube.com/watch?v=zVZgFzRGG50',
'https://youtube.com/watch?v=puhl4Od-4kg',
'https://youtube.com/watch?v=H8YWIwh0n9I',
'https://youtube.com/watch?v=2S4tJOK4xl0',
'https://youtube.com/watch?v=XQxZpkOqXnE',
'https://youtube.com/watch?v=EUpvk7VgClQ']


# i = 1
# for url in urls:
#     print(f'i={i}, url={url}')
#     i += 1
#     Download(url)

# Los Puentes 2021-04-23 Day
# playlistLink = 'https://www.youtube.com/playlist?list=PLq07ekK6H4qR9E0riiuyfEIHg0Npidd-F'
# cur_dir = 'video/Los Puentes 2021-04-23 Day'

# Tesoros de Kazan 2021-10-08 Day
# playlistLink = 'https://www.youtube.com/playlist?list=PLq07ekK6H4qRaCGjd1QxSASS3bLe7Im7f'
# cur_dir = 'video/Tesoros de Kazan 2021-10-08 Day'

# Tesoros de Kazan 2021-10-08 Evening
# playlistLink = 'https://www.youtube.com/playlist?list=PLq07ekK6H4qS_DXbYk0HIWPRPVuB0eGcb'
# cur_dir = 'video/Tesoros de Kazan 2021-10-08 Evening'

# Tesoros de Kazan 2021-10-09 Day
# playlistLink = 'https://www.youtube.com/playlist?list=PLq07ekK6H4qSAlzrsBBbCPzDmZuV_0Onz'
# cur_dir = 'video/Tesoros de Kazan 2021-10-09 Day'

# Tesoros de Kazan 2021-10-09 Evening with Tango en Vivo
# playlistLink = 'https://www.youtube.com/playlist?list=PLq07ekK6H4qSbzuo1iknk8dmnP65MmqRJ'
# cur_dir = 'video/Tesoros de Kazan 2021-10-09 Evening with Tango en Vivo'

# Tesoros de Kazan 2021-10-10 Day
# playlistLink = 'https://www.youtube.com/playlist?list=PLq07ekK6H4qQBRykjWhdy3JBqGBPoO88Z'
# cur_dir = 'video/Tesoros de Kazan 2021-10-10 Day'

# Vamos A Bailar Tver tango marathon 2018-10-27 Day
playlistLink = 'https://www.youtube.com/playlist?list=PLq07ekK6H4qSl_H6QqOipYkK6C2fOgxPq'
cur_dir = 'video/Vamos A Bailar Tver tango marathon 2018-10-27 Day'

# Vamos A Bailar Tver tango marathon 2018-10-26 Evening
# playlistLink = 'https://www.youtube.com/playlist?list=PLq07ekK6H4qSlGp5-s8cwsjih4Vc0Vlh7'
# cur_dir = 'video/Vamos A Bailar Tver tango marathon 2018-10-26 Evening'

# Vamos A Bailar Tver Tango marathon 2018-10-27 Evening
# playlistLink = 'https://www.youtube.com/playlist?list=PLq07ekK6H4qRWdfAy_t3KLqVJW0t52wwK'
# cur_dir = 'video/Vamos A Bailar Tver Tango marathon 2018-10-27 Evening'

# Vamos A Bailar Tver Tango marathon 2018-10-28 Day
playlistLink = 'https://www.youtube.com/playlist?list=PLq07ekK6H4qToC-KYCfG27OOXK1TNq5IW'
cur_dir = 'video/Vamos A Bailar Tver Tango marathon 2018-10-28 Day'

DownloadPlaylist(playlistLink, cur_dir)