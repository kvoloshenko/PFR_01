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
i = 1
for url in urls:
    print(f'i={i}, url={url}')
    i += 1
    Download(url)