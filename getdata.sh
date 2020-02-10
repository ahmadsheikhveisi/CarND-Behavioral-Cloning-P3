#!/bin/sh
# Get the data sets from googledrive

get_data()
{
FILEID=$1
FILENAME=$2
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILEID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$FILEID -O $FILENAME && rm -rf /tmp/cookies.txt

unzip $FILENAME -d /opt/carnd_p3
}

##

get_data '1sqvtFK9NtMRFoMZDNscLbGWwT8QH0wHF' '/opt/carnd_p3/Track1.zip'
get_data '1xOG1t-umD-MuqVNe4TBTFkv0Oe2Kd0Eb' '/opt/carnd_p3/Track1_reverse.zip'
get_data '1mAlDGBtx4fy4bttdg-6BMRy5MpHGVoZt' '/opt/carnd_p3/Curve1.zip'
