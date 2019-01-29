import wget
import urllib
import time

submission_data = False  # if False - it means we want to have comments data
years_to_get = [2011]#, 2014, 2013, 2012, 2011, 2010]
time_between_tries = 1*60
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
if submission_data:
    files_to_get = ['RS_v2_' + str(year) + '-' + m + '.xz' if year <= 2010 else 'RS_' + str(year) + '-'+m + '.bz2'
                    for year in years_to_get for m in months]
    pushshift_url = 'https://files.pushshift.io/reddit/submissions/'
else:
    files_to_get = ['RC_' + str(year) + '-' + m + '.xz' if year >= 2018 else 'RC_' + str(year) + '-'+m + '.bz2'
                    for year in years_to_get for m in months]
    pushshift_url = 'https://files.pushshift.io/reddit/comments/'


for f in files_to_get:
    full_path_to_file = pushshift_url + f
    got_result = False
    while got_result is not True:
        try:
            filename = wget.download(full_path_to_file)
            got_result = True
        except urllib.error.HTTPError:
            print('got "Too Many Requests" error, going to sleep a bit and will try again soon')
            time.sleep(time_between_tries)
