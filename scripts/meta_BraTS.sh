python /home/kdh/code/CLUE/meta/BraTS_meta.py t1 axial
python /home/kdh/code/CLUE/meta/BraTS_meta.py t1 coronal
python /home/kdh/code/CLUE/meta/BraTS_meta.py t1 sagittal
echo "T1 finished\n"

python /home/kdh/code/CLUE/meta/BraTS_meta.py t1ce axial
python /home/kdh/code/CLUE/meta/BraTS_meta.py t1ce coronal
python /home/kdh/code/CLUE/meta/BraTS_meta.py t1ce sagittal
echo "T1CE finished\n"

python /home/kdh/code/CLUE/meta/BraTS_meta.py t2 axial
python /home/kdh/code/CLUE/meta/BraTS_meta.py t2 coronal
python /home/kdh/code/CLUE/meta/BraTS_meta.py t2 sagittal
echo "T2 finished\n"

python /home/kdh/code/CLUE/meta/BraTS_meta.py flair axial
python /home/kdh/code/CLUE/meta/BraTS_meta.py flair coronal
python /home/kdh/code/CLUE/meta/BraTS_meta.py flair sagittal
echo "FLAIR finished\n"
