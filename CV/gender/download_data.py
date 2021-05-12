import os

os.system("mkdir data")
os.system("wget https://www.dropbox.com/s/zcwlujrtz3izcw8/gender.tgz data/")
os.system("tar xvzf gender.tgz -C data/")
os.system("rm gender.tgz")
