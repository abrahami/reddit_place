import pandas as pd
import zipfile
import sys
import hashlib
from base64 import b64encode

use_comments_data = True
# Data Loading
data_path = '/home/isabrah/reddit_data/' if sys.platform == 'linux' \
    else 'C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\PhD\\reddit canvas\\data\\'
zf = zipfile.ZipFile(data_path + 'tile_placements.zip')
tiles = pd.read_csv(zf.open('tile_placements.csv'))
tiles_users = set(tiles['user'])

if sys.platform == 'linux':
    submission_shrinked = pd.read_hdf(data_path + 'submission_shrinked.hdf')
    comments_shrinked = pd.read_hdf(data_path + 'comments_shrinked.hdf') if use_comments_data else None
    subreddits_df = pd.read_excel(data_path + 'all_subredditts_based_atlas_and_submissions.xlsx',
                                  sheet_name='Detailed list')

else:
    submission_shrinked = pd.read_hdf(data_path + 'sr_relations\\submission_shrinked.hdf')
    comments_shrinked = pd.read_hdf(data_path + 'sr_relations\\comments_shrinked.hdf') if use_comments_data else None
    subreddits_df = pd.read_excel(data_path + 'sr_relations\\all_subredditts_based_atlas_and_submissions.xlsx',
                                  sheet_name='Detailed list')

sub_users = set(submission_shrinked['author'])
com_users = set(comments_shrinked['author'])
writing_users = list(sub_users.union(com_users))

# hashing process
writing_hashed_users = []
for u in writing_users:
    m = hashlib.sha1()
    m.update(u.encode())
    u_digest = m.digest()
    writing_hashed_users.append((u, b64encode(u_digest).decode("utf-8")))

writing_hashed_users_set = set(u[1] for u in writing_hashed_users)
users_found_counter = 0
users_found = []
for u in tiles_users:
    if u in writing_hashed_users_set:
        users_found_counter += 1
        users_found.append(u)

print("% of users we were able to map using "
      "user's info is: {}".format(users_found_counter*1.0 / len(tiles_users)))

tiles_placements_found = sum([1 for u in list(tiles['user']) if u in writing_hashed_users_set])
print("% of tiles placements we were able to map using "
      "user's info is: {}".format(tiles_placements_found*1.0 / tiles.shape[0]))





