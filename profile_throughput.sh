FILE_NAME='throughput.log'
python profile_througpout.py --dataset yelp >> $FILE_NAME
python profile_througpout.py --dataset reddit >> $FILE_NAME
python profile_througpout.py --dataset ogbn-products >> $FILE_NAME
python profile_througpout.py --dataset ogbn-arxiv >> $FILE_NAME
python profile_througpout.py --dataset ogbn-papers100M >> $FILE_NAME
