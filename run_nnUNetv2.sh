DATASET_NAME_OR_ID="35"
DATASET_NAME_OR_ID_2="36"
DATASET_NAME_OR_ID_3="37"
DATASET_NAME_OR_ID_4="38"
TYPE="2d"

A="nohup nnUNetv2_train $DATASET_NAME_OR_ID $TYPE 0 --c"
B="nohup nnUNetv2_train $DATASET_NAME_OR_ID $TYPE 1 --c"
C="nohup nnUNetv2_train $DATASET_NAME_OR_ID $TYPE 2 --c"
D="nohup nnUNetv2_train $DATASET_NAME_OR_ID $TYPE 3 --c" 
E="nohup nnUNetv2_train $DATASET_NAME_OR_ID $TYPE 4 --c"

A2="nohup nnUNetv2_train $DATASET_NAME_OR_ID_2 $TYPE 0 --c"
B2="nohup nnUNetv2_train $DATASET_NAME_OR_ID_2 $TYPE 1 --c"
C2="nohup nnUNetv2_train $DATASET_NAME_OR_ID_2 $TYPE 2 --c"
D2="nohup nnUNetv2_train $DATASET_NAME_OR_ID_2 $TYPE 3 --c"
E2="nohup nnUNetv2_train $DATASET_NAME_OR_ID_2 $TYPE 4 --c"

A3="nohup nnUNetv2_train $DATASET_NAME_OR_ID_3 $TYPE 0 --c"
B3="nohup nnUNetv2_train $DATASET_NAME_OR_ID_3 $TYPE 1 --c"
C3="nohup nnUNetv2_train $DATASET_NAME_OR_ID_3 $TYPE 2 --c"
D3="nohup nnUNetv2_train $DATASET_NAME_OR_ID_3 $TYPE 3 --c"
E3="nohup nnUNetv2_train $DATASET_NAME_OR_ID_3 $TYPE 4 --c"

A4="nohup nnUNetv2_train $DATASET_NAME_OR_ID_4 $TYPE 0 --c"
B4="nohup nnUNetv2_train $DATASET_NAME_OR_ID_4 $TYPE 1 --c"
C4="nohup nnUNetv2_train $DATASET_NAME_OR_ID_4 $TYPE 2 --c"
D4="nohup nnUNetv2_train $DATASET_NAME_OR_ID_4 $TYPE 3 --c"
E4="nohup nnUNetv2_train $DATASET_NAME_OR_ID_4 $TYPE 4 --c"

$A && $B && $C && $D && $E && $A2 && $B2 && $C2 && $D2 && $E2 && $A3 && $B3 && $C3 && $D3 && $E3 && $A4 && $B4 && $C4 && $D4 && $E4   