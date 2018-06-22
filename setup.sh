
echo "---------------------------"
echo "Download data"
bash get_data.bash
echo "Done: Download data"
echo "---------------------------"

echo "---------code setup--------"
mkdir -p dataset
mkdir -p encoderdir
mkdir -p glove_vocab
mkdir -p logs
mkdir -p savedir
mkdir -p score
mkdir -p tuned_score
mkdir -p testlogdir

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
cd ..


# More data paraphrase effect capture
cd SICK_data
python script.py SICK/SICK_train.txt
python script.py SICK/SICK_trial.txt
python script.py SICK/SICK_test.txt
cd ..

mkdir -p dataset/SNLI_SICK
cat dataset/SNLI/s1.train SICK_data/s1.train >> dataset/SNLI_SICK/s1.train
cat dataset/SNLI/s2.train SICK_data/s2.train >> dataset/SNLI_SICK/s2.train
cat dataset/SNLI/labels.train SICK_data/labels.train >> dataset/SNLI_SICK/labels.train

cp dataset/SNLI/*.dev dataset/SNLI_SICK/
cp dataset/SNLI/*.test dataset/SNLI_SICK/

mkdir -p dataset/COMB_SICK
cat dataset/COMB/s1.train SICK_data/s1.train >> dataset/COMB_SICK/s1.train
cat dataset/COMB/s2.train SICK_data/s2.train >> dataset/COMB_SICK/s2.train
cat dataset/COMB/labels.train SICK_data/labels.train >> dataset/COMB_SICK/labels.train

cp dataset/COMB/*.dev dataset/COMB_SICK/
cp dataset/COMB/*.test dataset/COMB_SICK/



