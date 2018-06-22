
echo "---------------------------"
echo "Download data"
bash get_data.bash
echo "Done: Download data"
echo "---------------------------"

echo "---------code setup--------"
mkdir -p dataset
mv wmt_data/ dataset/
mv MultiNLI MNLI
mv MNLI/s1.dev.matched MNLI/s1.dev
mv MNLI/s2.dev.matched MNLI/s2.dev
mv MNLI/labels.dev.matched MNLI/labels.dev

mv MNLI/s1.dev.mismatched MNLI/s1.test
mv MNLI/s2.dev.mismatched MNLI/s2.test
mv MNLI/labels.dev.mismatched MNLI/labels.test

mv GloVe glove_dir
mv MNLI dataset/
mv SNLI dataset/

mkdir -p dataset/COMB
cd dataset
for split in train dev test
do
	fpath=SNLI/$split.snli.txt
	cat SNLI/s1.$split MNLI/s1.$split >> COMB/s1.$split
	cat SNLI/s2.$split MNLI/s2.$split >> COMB/s2.$split
	cat SNLI/labels.$split MNLI/labels.$split >> COMB/labels.$split
done

for dire in SNLI MNLI COMB
do
	sed -i -e 's/neutral/non-entailment/g' $dire/labels.train
	sed -i -e 's/neutral/non-entailment/g' $dire/labels.dev
	sed -i -e 's/neutral/non-entailment/g' $dire/labels.test

	sed -i -e 's/contradiction/non-entailment/g' $dire/labels.train
	sed -i -e 's/contradiction/non-entailment/g' $dire/labels.dev
	sed -i -e 's/contradiction/non-entailment/g' $dire/labels.test

done

