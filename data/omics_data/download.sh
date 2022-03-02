# FILEID can be obtained from the file google drive sharing link.
# FILEID='1-2q5xKyCBwZjwUdv1KwlSpCUJQ19vk9l'
# 1-1x1wxZGXCiKVuInRJu4HpdZG62GcB27
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-1x1wxZGXCiKVuInRJu4HpdZG62GcB27' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-1x1wxZGXCiKVuInRJu4HpdZG62GcB27" && rm -rf /tmp/cookies.txt