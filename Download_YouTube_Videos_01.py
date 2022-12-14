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

urls = ['https://youtu.be/OJRN1Lmwjuw',
        'https://youtu.be/UO1fTaVTClk',
        'https://youtu.be/UuBuojnty6o',
        'https://youtu.be/deabMSgWt4Q',
        'https://youtu.be/mCN1NLrSjiE',
        'https://youtu.be/YJ3Hn_F2diE',
        'https://youtu.be/CA3K-HUlg-A',
        'https://youtu.be/VSc8uIpiamw',
        'https://youtu.be/ryxicetVgB4',
        'https://youtu.be/eiqjnzmpiFE',
        'https://youtu.be/hQEKSSd18Zc',
        'https://youtu.be/L7uceNko_Ig',
        'https://youtu.be/tKZ4slDuKmE',
        'https://youtu.be/RXORG4S1KkA',
        'https://youtu.be/LBiCA5lCi_k']
i = 1
for url in urls:
    print(f'i={i}, url={url}')
    i += 1
    Download(url)