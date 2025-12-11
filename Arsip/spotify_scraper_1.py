import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time
import os
import re

# --- KONFIGURASI SPOTIFY ---
CLIENT_ID = '15f66381b91045c19c594d27bcbf58cc'
CLIENT_SECRET = '16fae6de3d1c4f16a2f7d7ba43bdc71e'

auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager, requests_timeout=10)

# --- NAMA FILE OUTPUT ---
file_name = "top_10_artists_songs_with_the_most_monthly_listeners_spotify.xlsx"

# --- FUNGSI BACA DATA EXISTING ---
def read_existing_data(file_name):
    if os.path.exists(file_name):
        return pd.read_excel(file_name)
    else:
        return pd.DataFrame(columns=['artis', 'judul_lagu', 'popularitas'])

# --- FUNGSI PEMBERSIHAN JUDUL LAGU ---
def clean_track_name(name):
    blacklist = [
        "remix", "live", "acoustic", "instrumental",
        "demo", "session", "version", "radio edit",
        "extended", "cover", "karaoke", "tribute"
    ]
    name = re.sub(r'\(.*?\)', '', name)   # hapus teks dalam ()
    name = re.sub(r'\[.*?\]', '', name)   # hapus teks dalam []
    name = re.sub(r'[^\w\s]', '', name)   # hapus simbol
    name = name.lower().strip()
    for word in blacklist:
        if word in name:
            return None
    return name

# --- DAFTAR ARTIS ---
artis_list = [
    "Bruno Mars", "The Weeknd", "Lady Gaga", "Billie Eilish",
    "Kendrick Lamar", "Coldplay", "Rihanna", "Ed Sheeran",
    "Ariana Grande", "Taylor Swift"
]

ARTIS_INDEX = 9
artis = artis_list[ARTIS_INDEX]

print(f"Sedang mencari lagu-lagu dari: {artis}")

existing_df = read_existing_data(file_name)

if not existing_df.empty and artis in existing_df['artis'].unique():
    print(f"Lagu dari {artis} sudah tersedia. Tidak perlu ambil lagi.")
    exit()

# --- CARI ARTIS DI SPOTIFY ---
result = sp.search(q=artis, type="artist", limit=1)
if not result['artists']['items']:
    print(f"Artis {artis} tidak ditemukan!")
    exit()

artist_id = result['artists']['items'][0]['id']

# --- AMBIL ALBUM ---
albums = []
results = sp.artist_albums(artist_id, album_type='album,single', limit=50)
albums.extend(results['items'])

while results['next']:
    results = sp.next(results)
    albums.extend(results['items'])

# --- FILTER ALBUM UNIK & BUKAN KOMPILASI ---
unique_album_names = set()
filtered_albums = []
for album in albums:
    if album['album_group'] == "compilation":
        continue
    cleaned_album_name = clean_track_name(album['name'])
    if cleaned_album_name and cleaned_album_name not in unique_album_names:
        unique_album_names.add(cleaned_album_name)
        filtered_albums.append(album)

print(f"{len(filtered_albums)} album unik ditemukan untuk {artis}")

MAX_TRACKS_PER_ALBUM = 20
seen_clean_titles = set()
songs_data = []

# --- LOOP SETIAP ALBUM ---
for album in filtered_albums:
    album_id = album['id']
    try:
        tracks_result = sp.album_tracks(album_id)
        tracks = tracks_result['items'][:MAX_TRACKS_PER_ALBUM]

        while tracks_result['next'] and len(tracks) < MAX_TRACKS_PER_ALBUM:
            tracks_result = sp.next(tracks_result)
            remaining = MAX_TRACKS_PER_ALBUM - len(tracks)
            tracks.extend(tracks_result['items'][:remaining])
    except Exception as e:
        print(f"Gagal mengambil track dari album {album_id}: {e}")
        continue

    for track in tracks:
        try:
            track_info = sp.track(track['id'])
            if not track_info:
                continue

            main_artist = track_info['artists'][0]['name'].lower()
            if artis.lower() not in main_artist:
                continue

            cleaned_title = clean_track_name(track_info['name'])
            if not cleaned_title or cleaned_title in seen_clean_titles:
                continue

            seen_clean_titles.add(cleaned_title)

            songs_data.append({
                'artis': artis,
                'judul_lagu': track_info['name'],
                'popularitas': track_info['popularity']
            })

            time.sleep(0.5)  # hindari rate limit

        except Exception as e:
            print(f"Gagal mengambil track {track['id']}: {e}")
            continue

# --- SIMPAN KE FILE ---
new_df = pd.DataFrame(songs_data)
combined_df = pd.concat([existing_df, new_df], ignore_index=True)
combined_df.to_excel(file_name, index=False)

print(f"Selesai mengambil data dari {artis}, total lagu baru: {len(new_df)}")