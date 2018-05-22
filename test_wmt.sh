# script for applying saved model once model is trained on SNLI data.

# bash test_wmt.sh $1 $2 --encoder_hidden_dim $3 --refpath $files --hyppath $files3 --lp $i

model_name=$1
dim=$2
file_name=$3
gpu=$4

mkdir -p logs/$3

ref_dir=../data/wmt_data/references
hyp_dir=../data/wmt_data/system-outputs/newstest2014


# remove files if already exists
sys_file=logs/$model_name/$model_name".sys.score"
if [ -f  $sys_file ] ; then
    rm $sys_file
fi

seg_file=logs/$model_name/$model_name".seg.score"
if [ -f  $seg_file ] ; then
    rm $seg_file
fi

declare -a arr=("cs-en" "de-en" "fr-en" "ru-en")

## now loop through the above array
for i in "${arr[@]}"; do
   echo "$i"
   for files in $ref_dir/*"$i";do
   		echo $files
   		a=2
   done

  	for files2 in $hyp_dir/*"$i";do
  		for files3 in $files2/*; do
  			sys_name=${files3_##*/}
  			echo $files2,$files3,$i
  			echo $files3,$sys_name

  			python test_wmt.py --outputmodelname $1 --enc_lstm_dim $2 --filename $3 --gpu_id $4 --refpath $files --hyppath $files3 --lp $i
  			#break
  		done
  	done
  	#break
done

mv logs/$3.desc logs/$3/
#mv logs/$3.sys.score logs/$3/
#mv logs/$3.seg.score logs/$3/