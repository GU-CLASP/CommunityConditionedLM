inputs := $(wildcard reddit_tokenized/*.txt)

all: $(inputs:reddit_tokenized/%.txt=reddit_splits/%.train.txt) $(inputs:reddit_tokenized/%.txt=reddit_splits/%.test.txt) $(inputs:reddit_tokenized/%.txt=reddit_splits/%.dev.txt) $(inputs:reddit_tokenized/%.txt=unigram_counts/%.count.txt) $(inputs:reddit_tokenized/%.txt=langid/%.test.txt) 
	echo "done"

reddit_shuff/%.txt: reddit_tokenized/%.txt
	mkdir -p reddit_shuff
	shuf $< | head -n 42000 > $@

reddit_splits/%.train.txt: reddit_shuff/%.txt
	mkdir -p reddit_splits
	tail -n 40000 $< > $@

reddit_splits/%.test-dev.txt: reddit_shuff/%.txt
	mkdir -p reddit_splits
	head -n 2000 $< > $@

reddit_splits/%.dev.txt: reddit_splits/%.test-dev.txt
	head -n 1000 $< > $@

reddit_splits/%.test.txt: reddit_splits/%.test-dev.txt
	tail -n 1000 $< > $@

unigram_counts/%.count.txt: reddit_splits/%.train.txt
	mkdir -p unigram_counts
	cat $< | tr ' ' '\n' | sort | uniq -c > $@

langid.py:
	wget https://raw.githubusercontent.com/saffsd/langid.py/master/langid/langid.py

langid/%.test.txt: langid.py reddit_splits/%.test.txt
	mkdir -p langid
	python3 -m langid --line < $< > $@

clean: 
	rm -rf reddit_shuff
	rm reddit_splits/*.test-dev.txt

