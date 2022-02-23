for f in data/reddit_splits/*.train.txt
do
	
	cat f | tr ' ' '\n' | sort | uniq -c | less
done
