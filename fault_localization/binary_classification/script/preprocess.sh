#/bin/bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate transfer
for i in InsertMissedStmt InsertNullPointerChecker MoveStmt  MutateConditionalExpr  MutateDataType  MutateLiteralExpr  MutateMethodInvExpr  MutateOperators  MutateReturnStmt  MutateVariable  RemoveBuggyStmt ;
do
    echo "--------------------------------"
    echo $i
    python data_preprocess.py $i
    if [ $? -ne 0 ]; then
        echo "[ERROR] "
        exit 1
    fi
done
